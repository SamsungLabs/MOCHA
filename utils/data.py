from torch.utils.data import DataLoader
import yaml

from datasets import PerSegDataset, EpisodicPerSeg, \
                     PODDataset, EpisodicPOD, \
                     CORe50Dataset, EpisodicCORe50, \
                     iCubWorldDataset, EpisodiciCubWorld

# this helper function initializes training 
# and validation set dataloaders depending on
# on the command line arguments
# See AuXFT codebase for more details
def get_train_val_loaders(args):
    if not args.episodic:
        if args.dataset == 'pod':
            tset = PODDataset(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['pod'],
                              coarse_labels=args.coarse_labels,
                              imgsz=672,
                              val=False,
                              augment=False)
            vset = PODDataset(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['pod'],
                              coarse_labels=args.coarse_labels,
                              imgsz=672,
                              val=True,
                              val_mode=args.val_mode,
                              augment=False)
        elif args.dataset == 'perseg':
            if not args.coarse_labels:
                raise ValueError('PerSeg Dataset can only be used in episodic mode')
            tset = PerSegDataset(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['perseg'],
                                 coarse_labels=args.coarse_labels,
                                 imgsz=672,
                                 val=False,
                                 augment=False)
            vset = None
        elif args.dataset == 'core50':
            if not args.coarse_labels and not args.debug:
                raise ValueError('CORe50 Dataset can only be used in episodic mode')
            tset = CORe50Dataset(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['core50'],
                                 coarse_labels=args.coarse_labels,
                                 imgsz=672,
                                 val=False,
                                 augment=False)
            vset = None
        elif args.dataset == 'icub':
            if not args.coarse_labels:
                raise ValueError('iCubWorld Dataset can only be used in episodic mode')
            tset = iCubWorldDataset(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['icubworld'],
                                    coarse_labels=args.coarse_labels,
                                    imgsz=672,
                                    val=False,
                                    augment=False)
            vset = None
        else:
            raise ValueError('Unrecognized dataset' + str(args.dataset))
    else:
        if args.dataset == 'pod':
            tset = EpisodicPOD(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['pod'],
                               coarse_labels=False,
                               imgsz=672,
                               val=False,
                               augment=False,
                               support=args.support,
                               cache_dataset=True)
            vset = EpisodicPOD(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['pod'],
                               coarse_labels=False,
                               imgsz=672,
                               val=True,
                               val_mode=args.val_mode,
                               augment=False,
                               cache_dataset=True)
        elif args.dataset == 'perseg':
            tset = EpisodicPerSeg(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['perseg'],
                                  coarse_labels=False,
                                  imgsz=672,
                                  val=False,
                                  augment=False,
                                  cache_dataset=True)
            vset = EpisodicPerSeg(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['perseg'],
                                  coarse_labels=False,
                                  imgsz=672,
                                  val=True,
                                  augment=False,
                                  cache_dataset=True)
        elif args.dataset == 'core50':
            tset = EpisodicCORe50(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['core50'],
                                  coarse_labels=False,
                                  imgsz=672,
                                  val=False,
                                  augment=False,
                                  support=args.support,
                                  cache_dataset=True)
            vset = EpisodicCORe50(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['core50'],
                                  coarse_labels=False,
                                  imgsz=672,
                                  val=True,
                                  augment=False,
                                  support=args.support,
                                  cache_dataset=True)
        elif args.dataset == 'icub':
            tset = EpisodiciCubWorld(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['icubworld'],
                                     coarse_labels=False,
                                     imgsz=672,
                                     val=False,
                                     augment=False,
                                     support=args.support,
                                     cache_dataset=True)
            vset = EpisodiciCubWorld(data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['icubworld'],
                                     coarse_labels=False,
                                     imgsz=672,
                                     val=True,
                                     augment=False,
                                     support=args.support,
                                     cache_dataset=True)
        else:
            raise ValueError('Unrecognized dataset' + str(args.dataset))

    tloader = DataLoader(tset,
                         8,
                         num_workers=0,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False,
                         collate_fn=tset.collate_fn)
    if vset is None:
        return tloader, None

    vloader = DataLoader(vset,
                         8,
                         num_workers=0,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False,
                         collate_fn=tset.collate_fn)

    return tloader, vloader
