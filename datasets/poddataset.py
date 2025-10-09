from copy import deepcopy
import json
import random
import yaml

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG

class PODDataset(YOLODataset):
    def __init__(self,
                 data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['pod'],
                 coarse_labels=True,
                 val=False,
                 val_mode=0,
                 augment=True,
                 path="datasets/poddataset.yaml",
                 use_segments=False,
                 use_keypoints=False,
                 **kwargs):
        assert val_mode in [0, 1, 2, 3, 4, 5], 'Illegal validation split'

        if val_mode == 0:
            vfolder = '/val'
        elif val_mode == 1:
            vfolder = '/val_light'
        elif val_mode == 2:
            vfolder = '/val_dark'
        elif val_mode == 3:
            vfolder = '/val_crop'
        elif val_mode == 4:
            vfolder = '/val_crop_light'
        else:
            vfolder = '/val_crop_dark'

        if val:
            img_path = data_path + vfolder
            augment = False
        else:
            img_path = data_path + '/train'
            augment = True and augment

        with open(path, encoding='utf-8') as fin:
            data = yaml.load(fin, yaml.BaseLoader)
        data['names'] = data['fine_names']

        self.data_path = path

        if coarse_labels:
            self.names = data['coarse_names']
        else:
            self.names = data['fine_names']

        self.nc = len(self.names)
        self.idmap = data['idmap']
        self.coarse_labels = coarse_labels
        self.valid_coarse = sorted([int(i) for i in set(self.idmap.values())])

        hyp = DEFAULT_CFG
        hyp.copy_paste = .5
        super().__init__(img_path=img_path, data=data, use_segments=use_segments, use_keypoints=use_keypoints, hyp=hyp, augment=augment, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.coarse_labels:
            for i in range(item['cls'].shape[0]):
                item['cls'][i] = int(self.idmap[str(item['cls'][i].int().item())])
        return item

class EpisodicPOD(PODDataset):
    def __init__(self,
                 data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['pod'],
                 coarse_labels=True,
                 val=False,
                 val_mode=0,
                 augment=True,
                 path="datasets/poddataset.yaml",
                 use_segments=False,
                 use_keypoints=False,
                 support=1,
                 cache_dataset=False,
                 **kwargs):
        super().__init__(data_path, coarse_labels, val, val_mode, augment, path, use_segments, use_keypoints, **kwargs)
        self.support = support
        self.val = val

        with open(data_path+'/class_to_images.json', encoding='utf-8') as fin:
            self.cfiles = json.load(fin)

        self.cache_dataset = cache_dataset
        if self.cache_dataset:
            self.cache = {}

        self.len = super().__len__()
        if not val:
            self.fnames = [f.replace('\\', '/').split('/')[-1].split('.')[0] for f in self.im_files]
            self.fids = {cl: {f: self.fnames.index(f) for f in v} for cl, v in self.cfiles.items()}
            self.init_episode()
            self.len = len(self.vids)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if not self.val:
            index = self.vids[index]

        if self.cache_dataset and index in self.cache:
            return deepcopy(self.cache[index])

        item = super().__getitem__(index)

        if self.cache_dataset:
            self.cache[index] = deepcopy(item)

        return item

    def init_episode(self, epid=None):
        if epid is not None:
            random.seed(epid)
        if self.val:
            return

        vids = []
        for cl in self.cfiles:
            if self.support <= len(self.cfiles[cl]):
                names = random.sample(self.cfiles[cl], self.support)
            else:
                names = self.cfiles[cl]
            vids += [self.fids[cl][n] for n in names]
        self.vids = vids
