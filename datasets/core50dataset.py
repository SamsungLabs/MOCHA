from copy import deepcopy
import json
import yaml

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG

class CORe50Dataset(YOLODataset):
    def __init__(self,
                 data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['core50'],
                 coarse_labels=True,
                 val=False,
                 augment=True,
                 path="datasets/core50dataset.yaml",
                 use_segments=False,
                 use_keypoints=False,
                 **kwargs):

        if val:
            img_path = data_path + '/val'
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

class EpisodicCORe50(CORe50Dataset):
    def __init__(self,
                 data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['core50'],
                 coarse_labels=True,
                 val=False,
                 augment=True,
                 path="datasets/core50dataset.yaml",
                 use_segments=False,
                 use_keypoints=False,
                 support=1,
                 cache_dataset=False,
                 **kwargs):
        super().__init__(data_path, coarse_labels, False, augment, path, use_segments, use_keypoints, **kwargs)
        self.val = val

        self.cache_dataset = cache_dataset
        if self.cache_dataset:
            self.cache = {}

        with open(data_path+'/episodes_%dshot.json'%support, encoding='utf-8') as fin:
            self.episodes = json.load(fin)
        self.epid = -1
        self.init_episode()

    def init_episode(self, epid=None):
        if epid is None:
            self.epid = (self.epid + 1) % len(self.episodes)
        else:
            self.epid = epid % len(self.episodes)

        fnames = [f.replace('\\', '/').split('/')[-1] for f in self.im_files]
        epfiles = self.episodes[self.epid]['val' if self.val else 'train']
        self.vids = [fnames.index(f) for f in epfiles]
        self.len = len(self.vids)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = self.vids[index]

        if self.cache_dataset and index in self.cache:
            return deepcopy(self.cache[index])

        item = super().__getitem__(index)

        if self.cache_dataset:
            self.cache[index] = deepcopy(item)

        return item
