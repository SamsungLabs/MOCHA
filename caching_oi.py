import random
import os
from time import time
from tqdm import tqdm
import numpy as np
from numpy import random as npr

import torch
from torch.utils.data import DataLoader, DistributedSampler

from datasets import OpenImages
from models import YoloFeats, LlavaClipVecs

def set_seed(seed):
    """
        sets the rng seeds for all libraries
    """
    torch.manual_seed(seed)
    npr.seed(seed)
    random.seed(seed)

def str2bool(s):
    """
        string to bool
    """
    s = s.lower()
    if s in ['1', 't', 'true']:
        return True
    if s in ['0', 'f', 'false']:
        return False
    raise ValueError(f"[{s}] cannot be parsed as boolean")

def init_loaders_and_models(rank, world_size, args):
    """
        initialize stuff here, to reduce SAM cost
    """
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

    yolof = YoloFeats(nc=tset.nc, is_base=True)

    odict = dict(yolof.state_dict())
    for k, v in torch.load(args.init_ckpt, map_location='cpu').items():
        if k.replace('module.', '') in odict:
            odict[k.replace('module.', '')] = v
        else:
            print('Ignoring key {%s}' % k.replace('module.', ''))
    yolof.load_state_dict(odict)
    yolof.to("cuda")
    for p in yolof.parameters():
        p.requires_grad = False
    yolof.eval()

    llavaclip = LlavaClipVecs(yolof, True)
    llavaclip.eval()

    return tloader, vloader, llavaclip

def store_feats(llavaclip, args, vloader):
    """
        do stuff here to reduce SAM cost
    """
    llavaclip.eval()

    stime = time()
    with torch.inference_mode():
        pbar = tqdm(vloader, desc=f'Storing Features', ncols=150)
        cache_path = os.path.join(args.cache_dir, 'llava-clip', 'openimages', 'val')
        cnames = list(vloader.dataset.names.values())
        for sample in pbar:
            if time() - stime > 60 * 9:  # 9 minutes
                break
            im_names = [name.replace('\\', '/').split('/')[-1].split('.')[0] \
                        for name in sample['im_file']]
            sample['names'] = cnames

            x = sample['img'] / 255.
            x = x.to('cuda', dtype=torch.float32)

            if all(os.path.isfile(os.path.join(cache_path, name + '.pth')) for name in im_names):
                llavaclipvecs = [
                    [(v.to(x.device), p, c) for (v, p, c) in \
                     torch.load(os.path.join(cache_path, name + '.pth'))
                     ] for name in im_names]
            else:
                llavaclipvecs, _ = llavaclip(x, conf=.3, sample=sample)
            for iname, name in enumerate(im_names):
                torch.save(llavaclipvecs[iname], os.path.join(cache_path, name + '.pth'))

def main(rank, world_size, args):
    """
        main function
    """
    set_seed(args.seed)
    tloader, vloader, llavaclip = init_loaders_and_models(rank, world_size, args)

    store_feats(llavaclip, args, tloader)
    store_feats(llavaclip, args, vloader)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_per_gpu", type=int, default=16)
    parser.add_argument("--init_ckpt", type=str, default="ckpts/base.pth")
    parser.add_argument("--disable_distributed", action='store_true')
    parser.add_argument('--cache_dir', type=str, default='cache/',
                        help="Cache where to cache the vectors for faster inference")
    g_args = parser.parse_args()

    if not g_args.disable_distributed:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        main(rank, world_size, g_args)
        torch.distributed.destroy_process_group()
    else:
        main(0, 1, g_args)
