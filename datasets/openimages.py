import yaml

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG

class OpenImages(YOLODataset):
    def __init__(self,
                 data_path=yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['openimages'],
                 val=False,
                 augment=True,
                 path="datasets/openimages.yaml",
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
        self.data_path = path
        self.names = data['names']
        self.nc = len(self.names)
        hyp = DEFAULT_CFG
        hyp.copy_paste = .5
        super().__init__(img_path=img_path, data=data, use_segments=use_segments, use_keypoints=use_keypoints, hyp=hyp, augment=augment, **kwargs)
