import random

import os
from shutil import rmtree
from time import time
from tqdm import tqdm
import numpy as np
from numpy import random as npr

import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
import torch.distributed as dist
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from datasets import OpenImages
from models import YoloFeats, ReconLoss, LlavaClipVecs, MochaFeatMap, YoloVecs, EmbeddingLoss
from utils import clean_predictions, Metrics

def set_seed(seed):
    
    torch.manual_seed(seed)
    npr.seed(seed)
    random.seed(seed)

def str2bool(s):
    
    s = s.lower()
    if s in ['1', 't', 'true']:
        return True
    if s in ['0', 'f', 'false']:
        return False
    raise ValueError(f"[{s}] cannot be parsed as boolean")

@torch.no_grad()
def clean_gradient(grad):
    grad[torch.isnan(grad)] = 0.
    grad[torch.isinf(grad)] = 0.
    return grad

def init_loaders_and_models(rank, world_size, args):
    
    tset = OpenImages(imgsz=672, augment=False)
    vset = OpenImages(imgsz=672, val=True)

    if not args.disable_distributed:
        tsampler = DistributedSampler(tset, num_replicas=world_size, rank=rank, shuffle=True)
        tloader = DataLoader(tset,
                             args.batch_per_gpu,
                             num_workers=16,
                             pin_memory=True,
                             drop_last=True,
                             sampler=tsampler,
                             collate_fn=tset.collate_fn)
    else:
        tloader = DataLoader(tset,
                             args.batch_per_gpu,
                             num_workers=16,
                             pin_memory=True,
                             drop_last=True,
                             shuffle=True,
                             collate_fn=tset.collate_fn)

    vloader = DataLoader(vset,
                         args.batch_per_gpu,
                         num_workers=8,
                         pin_memory=True,
                         drop_last=False,
                         shuffle=False,
                         collate_fn=tset.collate_fn)

    yolo = YoloVecs(nc=tset.nc,
                    use_gt_boxes=True,
                    is_base=True,
                    concatenate_chs=True,
                    dataset=tset,
                    pool_mode='mean')

    if args.init_ckpt != 'none':
        odict = dict(yolo.state_dict()) # dict() needed to silence pylint bug
        for k, v in torch.load(args.init_ckpt, map_location='cpu').items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        yolo.load_state_dict(odict)
    yolo.to("cuda")
    yolo.train()
    if not args.disable_distributed:
        yolo = DDP(yolo, device_ids=[rank])
    else:
        yolo = DataParallel(yolo, device_ids=[rank])


    for n, p in yolo.named_parameters():
        if '.dfl' in n:
            p.requires_grad = False
        if p.requires_grad:
            p.register_hook(clean_gradient)

    yolof = YoloFeats(nc=tset.nc,
                      is_base=True)

    odict = dict(yolof.state_dict()) # dict() needed to silence pylint bug
    for k, v in torch.load(args.init_ckpt, map_location='cpu').items():
        if k.replace('module.', '') in odict:
            odict[k.replace('module.', '')] = v
        else:
            print('Ignoring key {%s}'%k.replace('module.', ''))
    yolof.load_state_dict(odict)
    yolof.to("cuda")
    for p in yolof.parameters():
        p.requires_grad = False
    yolof.eval()

    llavaclip = LlavaClipVecs(yolof, True)
    llavaclip.eval()

    fmap = MochaFeatMap(outchs=args.pca_dim)
    fmap.to('cuda')
    fmap.train()
    if not args.disable_distributed:
        fmap = DDP(fmap, device_ids=[rank])
    else:
        fmap = DataParallel(fmap, device_ids=[rank])

    for p in fmap.parameters():
        p.register_hook(clean_gradient)

    return tloader, vloader, yolo, llavaclip, fmap

def init_losses_and_optim(args, fmap, yolo, tloader):
    
    rec = ReconLoss()
    emb = EmbeddingLoss()
    det = yolo.module.init_criterion()

    param_groups = [
        {'lr': args.lr, 'params':
            [p for p in fmap.parameters()
                if p.requires_grad],
            'weight_decay': args.wd},
        {'lr': args.lr/10, 'params':
            [p for n, p in yolo.named_parameters()
                if p.requires_grad and 'model.22.' in n],
            'weight_decay': 0}, # detection heads
        {'lr': args.lr/10, 'params':
            [p for n, p in yolo.named_parameters()
                if p.requires_grad and 'model.22.' not in n],
            'weight_decay': args.wd/10} # yolo backbone
    ]

    optim = Adam(param_groups, betas=(0.937, 0.999))
    scheduler = lr_scheduler.ChainedScheduler([
        lr_scheduler.LinearLR(optim, 1e-6, 1, args.warmup),
        lr_scheduler.CosineAnnealingLR(optim, args.epochs*len(tloader))
    ])
    return rec, emb, det, optim, scheduler

def train_log(writer, l, lrec, lemb, it, optim, lod=None, box=None, cls=None, dfl=None):
    
    writer.add_scalar('train/lr', optim.param_groups[0]['lr'], it)

    writer.add_scalar('train/ltot', l.item(), it)
    writer.add_scalar('train/lrec', lrec.item(), it)
    writer.add_scalar('train/lemb', lemb.item(), it)
    writer.add_scalar('train/grad', torch.stack(
            [p.grad.norm() for p in optim.param_groups[0]['params']]
        ).mean().item(), it)

    if box is not None and cls is not None and dfl is not None:
        writer.add_scalar('train/lod', lod.item(), it)
        writer.add_scalar('train/yolo/box', box.item(), it)
        writer.add_scalar('train/yolo/cls', cls.item(), it)
        writer.add_scalar('train/yolo/dfl', dfl.item(), it)


def train_distill_epoch(tloader, e, args, rank, optim, yolo, llavaclip,
                        fmap, pca_mean, pca_comp, rec, emb, scheduler, writer, it):
    
    yolo.eval()
    llavaclip.eval()
    fmap.train()

    if not args.disable_distributed:
        # set the current epoch, otherwise same order will be used each time
        tloader.sampler.set_epoch(e)

    cache_path = os.path.join(args.cache_dir, 'llava-clip', 'openimages', 'train')
    cnames = list(tloader.dataset.names.values())
    for iit, sample in enumerate(tqdm(tloader,
                                desc=f'Training Distillation Epoch [{e+1:03d}/{args.epochs:03d}]',
                                disable=rank>0, ncols=150)):
        im_names = [name.replace('\\', '/').split('/')[-1].split('.')[0] \
                    for name in sample['im_file']]
        sample['names'] = cnames

        optim.zero_grad()
        x = sample['img'] / 255.
        x = x.to('cuda', dtype=torch.float32)

        with torch.no_grad():
            yolovecs, _ = yolo(x, conf=.3, sample=sample)
            if all(os.path.isfile(os.path.join(cache_path, name+'.pth')) for name in im_names):
                llavaclipvecs = [
                    [(v.to(x.device), p, c) for (v, p, c) in \
                                torch.load(os.path.join(cache_path, name+'.pth'))
                    ] for name in im_names]
            else:
                llavaclipvecs, _ = llavaclip(x, conf=.3, sample=sample)
            if args.pca_dim > 0:
                llavaclipvecs = [[((v - pca_mean) @ pca_comp.T, p, c) \
                                  for (v, p, c) in box] for box in llavaclipvecs]
                llavaclipvecs = [[(fmap.module.normalize_pca_vec(v), p, c) \
                                  for (v, p, c) in box] for box in llavaclipvecs]

        x_v = torch.stack([fmap(v) for box in yolovecs for (v, _, _) in box])
        y_v = torch.stack([v for box in llavaclipvecs for (v, _, _) in box])
        lrec = rec(x_v, y_v)
        lemb = emb(x_v, y_v)
        l = lrec + args.lambda_emb * lemb
        l.backward()

        if rank == 0:
            train_log(writer, l, lrec, lemb, it, optim)

        optim.step()
        scheduler.step()
        it += 1
        if args.debug and iit > 10:
            break
    return it

def train_detection_epoch(tloader, e, args, rank, optim, yolo, llavaclip,
                          fmap, pca_mean, pca_comp, rec, emb, det, scheduler, writer, it):
    
    yolo.train()
    llavaclip.eval()
    fmap.train()


    if not args.disable_distributed:
        # set the current epoch, otherwise same order will be used each time
        tloader.sampler.set_epoch(e)

    cache_path = os.path.join(args.cache_dir, 'llava-clip', 'openimages', 'train')
    cnames = list(tloader.dataset.names.values())
    for iit, sample in enumerate(tqdm(tloader,
                                    desc=f'Training Detection Epoch [{e+1:03d}/{args.epochs:03d}]',
                                    disable=rank>0, ncols=150)):
        im_names = [name.replace('\\', '/').split('/')[-1].split('.')[0] \
                    for name in sample['im_file']]
        sample['names'] = cnames

        optim.zero_grad()
        x = sample['img'] / 255.
        x = x.to('cuda', dtype=torch.float32)

        # yolo input: simple normalization in 0-1
        pfeats, _, yolovecs = yolo.module.all_fw(x, conf=.3, sample=sample)

        # yolo detection loss
        yo, (box, cls, dfl) = det(pfeats, sample)
        lod = yo/args.batch_per_gpu

        with torch.no_grad():
            if all(os.path.isfile(os.path.join(cache_path, name+'.pth')) for name in im_names):
                llavaclipvecs = [
                    [(v.to(x.device), p, c) for (v, p, c) in \
                                torch.load(os.path.join(cache_path, name+'.pth'))
                    ] for name in im_names]
            else:
                llavaclipvecs, _ = llavaclip(x, conf=.3, sample=sample)
            if args.pca_dim > 0:
                llavaclipvecs = [[((v - pca_mean) @ pca_comp.T, p, c) \
                                  for (v, p, c) in box] for box in llavaclipvecs]
                llavaclipvecs = [[(fmap.module.normalize_pca_vec(v), p, c) \
                                  for (v, p, c) in box] for box in llavaclipvecs]

        x_v = torch.stack([fmap(v) for box in yolovecs for (v, _, _) in box])
        y_v = torch.stack([v for box in llavaclipvecs for (v, _, _) in box])
        lrec = rec(x_v, y_v)
        lemb = emb(x_v, y_v)
        lkd = lrec + args.lambda_emb * lemb

        l = lod + lkd
        l.backward()

        if rank == 0:
            train_log(writer, l, lrec, lemb, it, optim, lod, box, cls, dfl)

        optim.step()
        scheduler.step()
        it += 1
        if args.debug and iit > 10:
            break
    return it

def eval_epoch(yolo, llavaclip, fmap, pca_mean, pca_comp,
                args, vloader, e, rank, rec, emb, det, writer):
    
    yolo.eval()
    llavaclip.eval()
    fmap.eval()

    metrics = Metrics(vloader.dataset.names, conf=0.001)
    alkd, alod, abox, acls, adfl = 0, 0, 0, 0, 0

    stime = time()
    with torch.inference_mode():
        pbar = tqdm(vloader,
                    desc=f'Validation Epoch [{e+1:03d}/{args.epochs:03d}]',
                    disable=rank>0, ncols=150)
        it = 0
        cache_path = os.path.join(args.cache_dir, 'llava-clip', 'openimages', 'val')
        cnames = list(vloader.dataset.names.values())
        for it, sample in enumerate(pbar):
            if time() - stime > 60*9: # 9 minutes
                break
            im_names = [name.replace('\\', '/').split('/')[-1].split('.')[0] \
                        for name in sample['im_file']]
            sample['names'] = cnames

            x = sample['img'] / 255.
            x = x.to('cuda', dtype=torch.float32)

            (preds, pfeats), _, yolovecs = yolo.module.all_fw(x, conf=.3, sample=sample)
            boxes = yolo.module.get_results(preds)
            for i, box in enumerate(boxes):
                box, labels, cls = clean_predictions(box, sample, i)
                metrics(box, labels, cls)
            yo, (box, cls, dfl) = det(pfeats, sample)
            alod += yo/args.batch_per_gpu
            abox += box
            acls += cls
            adfl += dfl

            if all(os.path.isfile(os.path.join(cache_path, name+'.pth')) for name in im_names):
                llavaclipvecs = [
                    [(v.to(x.device), p, c) for (v, p, c) in \
                                torch.load(os.path.join(cache_path, name+'.pth'))
                    ] for name in im_names]
            else:
                llavaclipvecs, _ = llavaclip(x, conf=.3, sample=sample)
            if args.pca_dim > 0:
                llavaclipvecs = [[((v - pca_mean) @ pca_comp.T, p, c) \
                                  for (v, p, c) in box] for box in llavaclipvecs]
                llavaclipvecs = [[(fmap.module.normalize_pca_vec(v), p, c) \
                                  for (v, p, c) in box] for box in llavaclipvecs]

            x_v = torch.stack([fmap(v) for box in yolovecs for (v, _, _) in box])
            y_v = torch.stack([v for box in llavaclipvecs for (v, _, _) in box])
            alkd += rec(x_v, y_v) + args.lambda_emb * emb(x_v, y_v)

            if args.debug and it > 10:
                break

        lkd = alkd.item()/(it+1)
        _, _, map50_95 = metrics.get_ap()

        writer.add_scalar("val/lkd", lkd, e+1)
        writer.add_scalar("val/map", map50_95, e+1)

        writer.add_scalar("val/box", abox.item()/(it+1), e+1)
        writer.add_scalar("val/cls", acls.item()/(it+1), e+1)
        writer.add_scalar("val/dfl", adfl.item()/(it+1), e+1)
    return lkd, map50_95

def main(rank, world_size, args):
    

    # select the correct cuda device
    torch.cuda.set_device(rank)

    if not args.disable_distributed:
        # initialize the process group
        dist_url = "env://"
        print(f"| distributed init (rank {rank}): {dist_url}", flush=True)
        dist.init_process_group("nccl",
                                rank=rank,
                                init_method=dist_url,
                                world_size=world_size)

        dist.barrier()
    else:
        print("| running without distributed", flush=True)

    set_seed(args.seed)

    if rank == 0:
        rmtree(args.logdir, ignore_errors=True)
        writer = SummaryWriter(args.logdir, flush_secs=0.5)

        # extra initializations that need first run on master process
        _ = OpenImages(imgsz=672)
        _ = OpenImages(imgsz=672, val=True)
        llava = LlavaClipVecs(torch.nn.Module(), True)
        del llava
        torch.cuda.empty_cache()
    else:
        writer = None

    if not args.disable_distributed:
        dist.barrier()

    tloader, vloader, yolo, llavaclip, fmap = init_loaders_and_models(rank, world_size, args)
    rec, emb, det, optim, scheduler = init_losses_and_optim(args, fmap, yolo, tloader)

    if args.pca_dim > 0:
        pca_mean = args.pca_mean.to('cuda', torch.float32)
        pca_comp = args.pca_comp.to('cuda', torch.float32)
    else:
        pca_mean = None
        pca_comp = None

    bloss = float('inf')
    bap = 0
    it = 0
    for e in range(args.epochs):

        if (e + 1) % 4 == 0:
            it = train_detection_epoch(tloader, e, args, rank, optim, yolo, llavaclip,
                            fmap, pca_mean, pca_comp, rec, emb, det, scheduler, writer, it)
        else:
            it = train_distill_epoch(tloader, e, args, rank, optim, yolo, llavaclip,
                            fmap, pca_mean, pca_comp, rec, emb, scheduler, writer, it)

        if not args.disable_distributed:
            dist.barrier()

        if rank == 0:
            torch.save(fmap.state_dict(), args.logdir+'/latest.pth')
            lavg, map50_95 = eval_epoch(yolo, llavaclip, fmap, pca_mean, pca_comp,
                                    args, vloader, e, rank, rec, emb, det, writer)

            if bloss > lavg:
                bloss = lavg
                torch.save(fmap.state_dict(), args.logdir+'/best_kd.pth')

            if bap > map50_95:
                bap = map50_95
                torch.save(fmap.state_dict(), args.logdir+'/best_map.pth')

    if rank == 0:
        torch.save(fmap.state_dict(), args.logdir+'/final.pth')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_per_gpu", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--lambda_emb", type=float, default=1.)
    parser.add_argument("--use_kd", type=str2bool, default=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--logdir", type=str, default="logs/both")
    parser.add_argument("--init_ckpt", type=str, default="none")
    parser.add_argument("--disable_distributed", action='store_true')
    parser.add_argument('--cache_dir', type=str, default='cache/',
                        help="Cache where to cache the vectors for faster inference")
    parser.add_argument('--pca_path', type=str, default='ckpts/pca_oi.npz',
                        help="Path to the pre-computed pca npz file.")
    parser.add_argument('--pca_dim', type=int, default=256,
                        help="Whether to apply a dimensionality reduction via PCA, " + \
                        "<1: disabled. Applies only to the llava-clip model configuration.")
    g_args = parser.parse_args()

    if g_args.pca_dim > 0:
        with np.load(g_args.pca_path) as data:
            g_args.pca_mean = torch.tensor(data["mean"])
            g_args.pca_comp = torch.tensor(data["pca"][:g_args.pca_dim])

    if not g_args.disable_distributed:
        main(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), g_args)
    else:
        main(0, 1, g_args)

