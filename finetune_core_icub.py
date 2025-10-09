import random

from copy import deepcopy
from shutil import rmtree
from tqdm import tqdm
from numpy import random as npr

import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from datasets import CORe50Dataset, iCubWorldDataset
from models import YoloFeats
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

def init_loaders_and_models(rank, world_size, args):
    
    if args.dataset == "icub":
        tset = iCubWorldDataset(imgsz=672)
        vset = iCubWorldDataset(imgsz=672, augment=False)
    else:
        tset = CORe50Dataset(imgsz=672)
        vset = CORe50Dataset(imgsz=672, augment=False)

    tsampler = DistributedSampler(tset, num_replicas=world_size, rank=rank, shuffle=True)
    tloader = DataLoader(tset,
                         args.batch_per_gpu,
                         num_workers=16,
                         pin_memory=True,
                         drop_last=True,
                         sampler=tsampler,
                         collate_fn=tset.collate_fn)

    vloader = DataLoader(vset,
                         args.batch_per_gpu,
                         num_workers=8,
                         pin_memory=True,
                         drop_last=False,
                         shuffle=False,
                         collate_fn=tset.collate_fn)

    yolo = YoloFeats(nc=tset.nc, verbose=False)
    sdict = dict(yolo.state_dict()) #silence pylint bug
    ndict = torch.load(args.pretrained_ckpt, map_location='cpu')
    if 'model' in ndict:
        ndict = ndict['model'].state_dict()
        for k in sdict:
            if 'model.22.cv3' not in k and 'fmap' not in k:
                sdict[k] = ndict[k]
    else:
        sdict = {k.replace('module.', ''): v for k,v in ndict.items()}
    yolo.load_state_dict(sdict) # initialize yolo with distilled dino weights
    yolo.to('cuda')
    yolo.eval()
    yolo = DDP(yolo, device_ids=[rank], find_unused_parameters=True)

    ema_dict = deepcopy(yolo.state_dict())
    # freeze layers as in: https://github.com/ultralytics/
    #                      ultralytics/blob/main/ultralytics/engine/trainer.py#L214
    for n, p in yolo.named_parameters():
        if 'model.22.cv3' in n:
            print("Parameter", n, "is trainable.")
        else:
            p.requires_grad = False

    return vset, tloader, vloader, yolo, ema_dict

def init_losses_and_optim(args, yolo, tloader):
    
    det = yolo.module.init_criterion()

    param_groups = [
        {'lr': args.lr, 'params':
            [p for p in yolo.parameters() if p.requires_grad],
         'weight_decay': args.wd}
    ]

    optim = Adam(param_groups, betas=(0.937, 0.999))
    scheduler = lr_scheduler.ChainedScheduler([
        lr_scheduler.CosineAnnealingLR(optim, args.epochs*len(tloader))
    ])
    return det, optim, scheduler

def train_log(writer, l, it, optim, yo, box, cls, dfl):
    
    writer.add_scalar('train/ltot', l.item(), it)
    writer.add_scalar('train/lr/yolo', optim.param_groups[0]['lr'], it)
    writer.add_scalar('train/wd/yolo', optim.param_groups[0]['weight_decay'], it)
    writer.add_scalar('train/lr/dino', optim.param_groups[-1]['lr'], it)

    writer.add_scalar('train/yolo/tot', yo.item(), it)
    writer.add_scalar('train/yolo/box', box.item(), it)
    writer.add_scalar('train/yolo/cls', cls.item(), it)
    writer.add_scalar('train/yolo/dfl', dfl.item(), it)

def update_ema(it, args, yolo, ema_dict):
    
    # ema step and reset yolo
    if it % args.ema_step == 0:
        sdict = dict(yolo.state_dict()) # silence error again
        for k in ema_dict:
            ema_dict[k] = args.ema_rate*ema_dict[k] + (1-args.ema_rate)*sdict[k]
        yolo.load_state_dict(ema_dict)
    return yolo, ema_dict

def set_wd(optim, args):
    for pg in optim.param_groups:
        if pg['weight_decay'] > 0:
            pg['weight_decay'] = args.wd * pg['lr']/args.lr

def train_epoch(tloader, e, args, rank, optim, yolo, det, scheduler, ema_dict, writer, it):
    # set the current epoch, otherwise same order will be used each time
    tloader.sampler.set_epoch(e)
    for _, sample in enumerate(tqdm(tloader, desc='Training Epoch \
                [%03d/%03d]'%(e+1, args.epochs), disable=rank>0, ncols=150)):
        set_wd(optim, args)

        optim.zero_grad()
        x = sample['img'] / 255.
        x = x.to('cuda', dtype=torch.float32)

        # yolo input: simple normalization in 0-1
        pfeats, _ = yolo(x)

        # yolo detection loss
        yo, (box, cls, dfl) = det(pfeats, sample)
        l = yo/args.batch_per_gpu
        l.backward()

        if rank == 0:
            train_log(writer, l, it, optim, yo, box, cls, dfl)

        optim.step()
        scheduler.step()
        it += 1

        yolo, ema_dict = update_ema(it, args, yolo, ema_dict)
    return it

def eval_epoch(yolo, args, vset, vloader, e, rank, det, writer):
    torch.save(yolo.state_dict(), args.logdir+'/yolo_latest.pth')

    metrics = Metrics(vset.names, conf=0.001)
    ayo, abox, acls, adfl = 0, 0, 0, 0
    with torch.inference_mode():
        pbar = tqdm(vloader, desc='Validation Epoch [%03d/%03d], mAP50-90: \
                    %02.2f%%'%(e+1, args.epochs, 0), disable=rank>0, ncols=150)
        for _, sample in enumerate(pbar):
            x = sample['img'] / 255.
            x = x.to('cuda', dtype=torch.float32)

            # yolo input: simple normalization in 0-1
            (pred, pfeats), _ = yolo(x)

            boxes = yolo.module.get_results(pred)
            for i, box in enumerate(boxes):
                box, labels, cls = clean_predictions(box, sample, i)
                metrics(box, labels, cls)

            yo, (box, cls, dfl) = det(pfeats, sample)
            ayo += yo
            abox += box
            acls += cls
            adfl += dfl

            map50, map75, map50_95 = metrics.get_ap()
            pbar.set_description('Validation Epoch [%03d/%03d], mAP50-90: %02.2f%%'%(e+1, args.epochs, map50_95))

        writer.add_scalar('val/yolo/tot', ayo.item()/len(vloader), e+1)
        writer.add_scalar('val/yolo/box', abox.item()/len(vloader), e+1)
        writer.add_scalar('val/yolo/cls', acls.item()/len(vloader), e+1)
        writer.add_scalar('val/yolo/dfl', adfl.item()/len(vloader), e+1)

        writer.add_scalar('val/metrics/mAP50', map50, e+1)
        writer.add_scalar('val/metrics/mAP75', map75, e+1)
        writer.add_scalar('val/metrics/mAP50-95', map50_95, e+1)

    return map50_95

def main(rank, world_size, args):
    dist_url = "env://"

    # select the correct cuda device
    torch.cuda.set_device(rank)

    # initialize the process group
    print(f"| distributed init (rank {rank}): {dist_url}", flush=True)
    dist.init_process_group("nccl",
                            rank=rank,
                            init_method=dist_url,
                            world_size=world_size)
    dist.barrier()

    set_seed(args.seed)

    if rank == 0:
        rmtree(args.logdir, ignore_errors=True)
        writer = SummaryWriter(args.logdir, flush_secs=0.5)

        # extra initializations that need first run on master process
        if args.dataset == "icub":
            _ = iCubWorldDataset(imgsz=672)
            _ = iCubWorldDataset(imgsz=672, augment=False)
        else:
            _ = CORe50Dataset(imgsz=672)
            _ = CORe50Dataset(imgsz=672, augment=False)
    else:
        writer = None

    dist.barrier()

    vset, tloader, vloader, yolo, ema_dict = init_loaders_and_models(rank, world_size, args)
    det, optim, scheduler = init_losses_and_optim(args, yolo, tloader)

    bap = 0
    it = 0
    for e in range(args.epochs):
        yolo.eval()

        it = train_epoch(tloader, e, args, rank, optim, yolo, det, scheduler, ema_dict, writer, it)

        dist.barrier()

        yolo.eval()
        if rank == 0:

            map50_95 = eval_epoch(yolo, args, vset, vloader, e, rank, det, writer)

            if bap < map50_95:
                bap = map50_95
                torch.save(yolo.state_dict(), args.logdir+'/yolo_best.pth')

    if rank == 0:
        torch.save(yolo.state_dict(), args.logdir+'/yolo_final.pth')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_per_gpu", type=int, default=16)
    parser.add_argument("--ema_step", type=int, default=10)
    parser.add_argument("--ema_rate", type=float, default=.6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--pretrained_ckpt", default="ckpts/auxft.pth")
    parser.add_argument("--dataset", default="core50", choices=['core50', 'icub'])
    parser.add_argument("--logdir", type=str, default="logs/finetune")
    g_args = parser.parse_args()

    import os
    g_args.logdir = os.path.join(g_args.logdir, g_args.dataset).replace("\\", "/")
    main(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), g_args)
