from ultralytics.data.dataset import YOLODataset
import yaml

root = yaml.load(open('datasets/data_paths.yaml'), yaml.BaseLoader)['coco']
class COCO(YOLODataset):
    def __init__(self,
                 val=False,
                 path="datasets/coco.yaml",
                 use_segments=False,
                 use_keypoints=False,
                 **kwargs):
        if val:
            img_path = root + '/val'
            augment = False
        else:
            img_path = root + '/train'
            augment = True

        with open(path, encoding='utf-8') as fin:
            data = yaml.load(fin, yaml.BaseLoader)
        self.names = data['names']
        self.nc = len(self.names)
        super().__init__(img_path=img_path, data=data, use_segments=use_segments, use_keypoints=use_keypoints, augment=augment, **kwargs)
