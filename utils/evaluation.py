import numpy as np
import torch

from ultralytics.utils.metrics import ConfusionMatrix, box_iou, ap_per_class
from ultralytics.utils.ops import scale_boxes, xywh2xyxy

# See AuXFT codebase for more details
def clean_predictions(box, sample, i):
    idx = sample['batch_idx'] == i

    h0, w0 = sample['ori_shape'][i]
    h1, w1 = sample['img'][i].shape[1:]

    cls = sample['cls'][idx]

    box[:,:4] = scale_boxes(
                        (h1,w1),
                        box[:,:4],
                        (h0,w0),
                        sample['ratio_pad'][i]
                    )
    labels = scale_boxes(
                        (h1,w1),
                        xywh2xyxy(sample['bboxes'][idx])*torch.tensor((w1,h1,w1,h1)),
                        (h0,w0),
                        sample['ratio_pad'][i]
                    )
    labels = torch.cat((cls, labels), 1)
    return box[box[:, -1]>=0], labels.to(box.device), cls.squeeze(-1)

# See AuXFT codebase for more details
class Metrics():
    def __init__(self, names, conf):
        self.nc = len(names)
        self.names = names
        self.conf = conf

        self.cm = ConfusionMatrix(nc=self.nc, conf=self.conf)

        self.iouv = torch.linspace(0.5, 0.95, 10) # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

        self.stats = []

    def __call__(self, box, labels, cls):
        # aggregate metrics
        self.cm.process_batch(box, labels)

        # compute per-batch statistics
        iou = box_iou(labels[:, 1:], box[:, :4])
        correct_bboxes = self.match_predictions(box[:, 5], labels[:, 0], iou)
        self.stats.append((correct_bboxes, box[:, 4], box[:, 5], cls))

    def get_ap(self):
        stats = self.get_all_stats()
        if stats is None:
            return 0, 0, 0
        ap = 100*stats[5]
        map50 = ap[:,0].mean()
        map75 = ap[:,5].mean()
        map50_95 = ap.mean()
        return map50, map75, map50_95

    def get_map(self):
        tp, fp = self.cm.tp_fp()
        return 100*np.mean(tp/(tp+fp+1e-5))

    def get_iap(self):
        stats = self.get_all_stats()
        ap = 100*stats[5].mean(axis=1)
        return ap

    def print_ap(self, ap=None, std=None):
        stats = self.get_all_stats()
        if ap is None:
            ap = 100*stats[5].mean(axis=1)
        names = [self.names[str(cid)] for cid in stats[6]]
        if std is None:
            for name, p in zip(names, ap):
                print('% 20s \t %06.3f'%(name, p))
        else:
            for name, p, s in zip(names, ap, std):
                print('% 20s \t %06.3f +/- %06.3f'%(name, p, s))

    def get_all_stats(self):
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        if len(stats) and stats[0].any():
            return ap_per_class(*stats, names=self.names)
        return None

    def match_predictions(self, pred_classes, true_classes, iou):
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
