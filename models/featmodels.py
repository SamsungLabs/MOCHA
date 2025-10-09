from copy import deepcopy
from typing import Any, Mapping

from scipy.stats import skew

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.nn import DetectionModel
from ultralytics.nn.modules import Detect, Segment, Pose
from ultralytics.nn.tasks import yaml_model_load, parse_model
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, IterableSimpleNamespace
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import initialize_weights

from transformers import AutoProcessor, LlavaForConditionalGeneration, CLIPProcessor, CLIPModel

# Exension of the YOLO class to expose internal features
class YoloFeats(DetectionModel):
    def __init__(self,
                 cfg='yolov8n.yaml',
                 ch=3,
                 nc=None,
                 verbose=False,
                 use_fcn=False,
                 dinoc=384,
                 dinos=48,
                 is_base=False):

        super(DetectionModel, self).__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)

        self.overrides = {}
        self.overrides['model'] = cfg

        self.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, **self.overrides})

        self.dinoc = dinoc
        self.dinos = dinos
        self.nc = nc
        self.use_fcn = use_fcn
        self.is_base = is_base

        if not self.is_base:
            if self.use_fcn:
                self.fmap11 = nn.Conv2d( 64, 4*64, 3, padding=1, bias=False)
                self.fmap12 = nn.Conv2d(128, 4*128, 3, padding=1, bias=False)
                self.fmap13 = nn.Conv2d(256, 4*256, 3, padding=1, bias=False)

                self.fmap21 = nn.Conv2d(4*64, dinoc, 3, padding=1, bias=False)
                self.fmap22 = nn.Conv2d(4*128, dinoc, 3, padding=1, bias=False)
                self.fmap23 = nn.Conv2d(4*256, dinoc, 3, padding=1, bias=False)

                self.relu = nn.ReLU()
            else:
                self.fmap1 = nn.Conv2d( 64, dinoc, 3, padding=1, bias=False)
                self.fmap2 = nn.Conv2d(128, dinoc, 3, padding=1, bias=False)
                self.fmap3 = nn.Conv2d(256, dinoc, 3, padding=1, bias=False)

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info("Overriding model.yaml nc=%s with nc=%d", self.yaml['nc'], nc) # MODIFIED
            self.yaml['nc'] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0]
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def get_results(self, preds, conf=0.001, iou_thres=0.6):
        return non_max_suppression(preds,
                                   conf,
                                   iou_thres,
                                   agnostic=self.args.agnostic_nms,
                                   max_det=self.args.max_det,
                                   classes=self.args.classes)

    def _predict_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        fs = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if 'Detect' in m.__class__.__name__:
                fs = [xi for xi in x]
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        if self.is_base:
            return x, fs

        yf1, yf2, yf3 = fs
        if self.use_fcn:
            yf1 = self.fmap11(yf1)
            yf1 = self.fmap21(self.relu(yf1))

            yf2 = self.fmap12(yf2)
            yf2 = self.fmap22(self.relu(yf2))

            yf3 = self.fmap13(yf3)
            yf3 = self.fmap23(self.relu(yf3))
        else:
            yf1 = self.fmap1(yf1)
            yf2 = self.fmap2(yf2)
            yf3 = self.fmap3(yf3)

        yf1 = F.interpolate(yf1, (self.dinos, self.dinos), mode='area')                         # downsample
        yf2 = F.interpolate(yf2, (self.dinos, self.dinos), mode='bilinear', align_corners=True) # ~ resample (42x42 vs. 48x48)
        yf3 = F.interpolate(yf3, (self.dinos, self.dinos), mode='bicubic', align_corners=True)  # upsample

        return x, (yf1, yf2, yf3)

# See AuXFT codebase for more details
class ColMap(nn.Module):
    def __init__(self):
        super().__init__()

        self.fmap = nn.Conv2d(384, 3, 1, bias=False)
        self.ifmap = nn.Conv2d(3, 384, 1, bias=False)
        self.sigma = nn.Sigmoid()

    def forward(self, x):
        rgb = self.sigma(self.fmap(x))
        rx = self.ifmap(rgb)
        return rgb, rx

    def color_features(self, feat):
        with torch.no_grad():
            return self.sigma(self.fmap(feat))

# See AuXFT codebase for more details
class DinoFeats(nn.Module):
    def __init__(self,
                 yoloc=(64,128,256),
                 yolos=(28, 14, 7),
                 mode='conv'):
        super().__init__()

        self.yoloc = yoloc
        self.yolos = yolos
        self.mode = mode

        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False

        self.colmap = ColMap()

    def train(self, *args, **kwargs):
        o = super().train(*args, **kwargs)
        self.dino.eval()
        return o

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        o = super().load_state_dict(state_dict, strict, assign)
        # reset dino for safety
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False
        return o

    def forward(self, x):
        B, _, H, W = x.shape
        H1, W1 = H//14, W//14
        with torch.no_grad():
            o = self.dino(x, is_training=True)["x_prenorm"]
            #x, c = o["x_prenorm"], o['x_norm_clstoken']
            x, c = o[:, self.dino.num_register_tokens + 1 :], o[:,self.dino.num_register_tokens]
            x = x.permute(0,2,1).reshape(B,384,H1,W1)

        _, rx = self.colmap(x)
        return x, rx, c

    def color_features(self, feat):
        
        return self.colmap.color_features(feat)

# Exension of YoloFeats that performs AvgPool based on b_i
# (Paper Methodology Section, Eq. 4)
class YoloVecs(YoloFeats):
    def __init__(self,
                 cfg='yolov8n.yaml',
                 ch=3,
                 nc=None,
                 verbose=False,
                 use_gt_boxes=False,
                 use_fcn=False,
                 dinoc=384,
                 dinos=48,
                 is_base=False,
                 concatenate_chs=True,
                 mask_extra_coarse=False,
                 dataset=None,
                 pool_mode='mean'):
        self.mask_extra_coarse = mask_extra_coarse

        if mask_extra_coarse:
            assert dataset is not None, "A dataset object is needed to mask the extra coarse classes"
            self.ids_to_mask = [i+4 for i in range(nc) if i not in dataset.valid_coarse] # shift indices to class channels

        super().__init__(cfg, ch, nc, verbose, use_fcn, dinoc, dinos, is_base)
        self.use_gt_boxes = use_gt_boxes
        self.concatenate_chs = concatenate_chs
        self.pool_mode = pool_mode

    def __call__(self, x, conf=.3, sample=None) -> Any:
        assert not self.training, "To use this model only works in evaluation mode"
        (pred, _), feats =  super().__call__(x)

        if self.mask_extra_coarse:
            pred[:,self.ids_to_mask] = 0

        if not self.is_base:
            feat = feats[0] + feats[1] + feats[2]
        if self.use_gt_boxes and sample is not None:
            boxlist = []
            for i in range(x.shape[0]):
                H1, W1 = sample['img'][i].shape[1:]
                idx = sample['batch_idx'] == i
                box = xywh2xyxy(sample['bboxes'][idx]).to(device=x.device)*torch.tensor((W1,H1,W1,H1), device=x.device)
                boxlist.append(torch.cat([
                                    box.to(device=x.device),
                                    torch.ones_like(sample['cls'][idx]).to(device=x.device),
                                    sample['cls'][idx].to(device=x.device)
                                ], dim=1))

        else:
            boxlist = self.get_results(pred, conf=conf)

        return self.cut_features_base(boxlist, feats, x.shape[2:]) if self.is_base \
                    else self.cut_features(boxlist, feat, x.shape[2:]), boxlist # pylint: disable=all

    def cut_features(self, boxlist, feat, oshape):
        batch = []
        for i, boxes in enumerate(boxlist):
            obox = []
            for box in boxes:
                conf, clas = box[4].item(), int(box[5])

                fshape = feat.shape[2:]
                scale = torch.tensor([fshape[0]/oshape[0], fshape[1]/oshape[1], fshape[0]/oshape[0], fshape[1]/oshape[1]], device=feat.device)
                coords = box[:4]*scale
                coords[[0,1]], coords[[2,3]] = torch.floor(coords[[0,1]]), torch.ceil(coords[[2,3]])
                coords = torch.clamp_min(coords.int(), 0)

                cut = feat[i,:,coords[1]:coords[3]+1, coords[0]:coords[2]+1].reshape(feat.shape[1],-1)
                if self.pool_mode == 'mean':
                    obox.append([cut.mean(dim=-1), conf, clas])
                elif self.pool_mode == 'max':
                    obox.append([cut.max(dim=-1)[0], conf, clas])
                elif self.pool_mode == 'median':
                    obox.append([cut.median(dim=-1)[0], conf, clas])
                elif self.pool_mode == 'std':
                    obox.append([torch.cat([cut.mean(dim=-1), cut.std(dim=-1)]), conf, clas])
                elif self.pool_mode == 'skew':
                    obox.append([torch.cat([cut.mean(dim=-1), cut.std(dim=-1), torch.from_numpy(skew(cut.cpu().numpy(), axis=-1)).to(device=cut.device)]), conf, clas])
                else:
                    raise ValueError('Unknown pooling strategy {%s}'%self.pool_mode)

            batch.append(obox)
        return batch

    def cut_features_base(self, boxlist, feats, oshape):
        batch = []
        for i, boxes in enumerate(boxlist):
            obox = []
            for box in boxes:
                conf, clas = box[4].item(), int(box[5])
                if self.concatenate_chs:
                    ofeat = []
                    for feat in feats:
                        fshape = feat.shape[2:]
                        scale = torch.tensor([fshape[0]/oshape[0], fshape[1]/oshape[1], fshape[0]/oshape[0], fshape[1]/oshape[1]], device=feat.device)
                        coords = box[:4]*scale
                        coords[[0,1]], coords[[2,3]] = torch.floor(coords[[0,1]]), torch.ceil(coords[[2,3]])
                        coords = torch.clamp_min(coords.int(), 0)

                        cut = feat[i,:,coords[1]:coords[3]+1, coords[0]:coords[2]+1].reshape(feat.shape[1],-1)
                        if self.pool_mode == 'mean':
                            ofeat.append(cut.mean(dim=-1))
                        elif self.pool_mode == 'max':
                            ofeat.append(cut.max(dim=-1)[0])
                        elif self.pool_mode == 'median':
                            ofeat.append(cut.median(dim=-1)[0])
                        elif self.pool_mode == 'std':
                            ofeat.append(torch.cat([cut.mean(dim=-1), cut.std(dim=-1)]))
                        elif self.pool_mode == 'skew':
                            ofeat.append(torch.cat([cut.mean(dim=-1), cut.std(dim=-1), torch.from_numpy(skew(cut.cpu().numpy(), axis=-1)).to(device=cut.device)]))
                        else:
                            raise ValueError('Unknown pooling strategy {%s}'%self.pool_mode)

                    obox.append([torch.cat(ofeat, dim=0), conf, clas])
                else:
                    feat = feats[2]
                    fshape = feat.shape[2:]
                    scale = torch.tensor([fshape[0]/oshape[0], fshape[1]/oshape[1], fshape[0]/oshape[0], fshape[1]/oshape[1]], device=feat.device)
                    coords = box[:4]*scale
                    coords[[0,1]], coords[[2,3]] = torch.floor(coords[[0,1]]), torch.ceil(coords[[2,3]])
                    coords = torch.clamp_min(coords.int(), 0)

                    cut = feat[i,:,coords[1]:coords[3]+1, coords[0]:coords[2]+1].reshape(feat.shape[1],-1)
                    if self.pool_mode == 'mean':
                        obox.append([cut.mean(dim=-1), conf, clas])
                    elif self.pool_mode == 'max':
                        obox.append([cut.max(dim=-1)[0], conf, clas])
                    elif self.pool_mode == 'median':
                        obox.append([cut.median(dim=-1)[0], conf, clas])
                    elif self.pool_mode == 'std':
                        obox.append([torch.cat([cut.mean(dim=-1), cut.std(dim=-1)]), conf, clas])
                    elif self.pool_mode == 'skew':
                        obox.append([torch.cat([cut.mean(dim=-1), cut.std(dim=-1), torch.from_numpy(skew(cut.cpu().numpy(), axis=-1)).to(device=cut.device)]), conf, clas])
                    else:
                        raise ValueError('Unknown pooling strategy {%s}'%self.pool_mode)
            batch.append(obox)
        return batch

    def all_fw(self, x, conf=.3, sample=None):
        if self.training:
            pred, feats =  super().__call__(x)
        else:
            (pred, pfeats), feats = super().__call__(x)

        if self.mask_extra_coarse:
            pred[:,self.ids_to_mask] = 0

        if self.use_gt_boxes and sample is not None:
            boxlist = []
            for i in range(x.shape[0]):
                H1, W1 = sample['img'][i].shape[1:]
                idx = sample['batch_idx'] == i
                box = xywh2xyxy(sample['bboxes'][idx])*torch.tensor((W1,H1,W1,H1))
                boxlist.append(torch.cat([box, torch.ones_like(sample['cls'][idx]), sample['cls'][idx]], dim=1).to(device=x.device))

        else:
            boxlist = self.get_results(pred, conf=conf)

        if self.training:
            return pred, feats, self.cut_features_base(boxlist, feats, x.shape[2:])
        else:
            return (pred, pfeats), feats, self.cut_features_base(boxlist, feats, x.shape[2:])

# See AuXFT codebase for more details
class DinoVecs(nn.Module):
    def __init__(self,
                 yolo,
                 use_gt_boxes=False):
        super().__init__()

        self.use_gt_boxes = use_gt_boxes

        self.yolo = yolo
        self.yolo.eval()

        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dino.eval()

    def forward(self, x, conf=.3, sample=None):
        B, _, H, W = x.shape
        H1, W1 = H//14, W//14

        (pred, _), _ = self.yolo(x)
        feats = self.dino(x, is_training=True)["x_norm_patchtokens"]
        feats = feats.permute(0,2,1).reshape(B,384,H1,W1)

        if self.use_gt_boxes and sample is not None:
            boxlist = []
            for i in range(x.shape[0]):
                H1, W1 = sample['img'][i].shape[1:]
                idx = sample['batch_idx'] == i
                box = xywh2xyxy(sample['bboxes'][idx])*torch.tensor((W1,H1,W1,H1))
                boxlist.append(torch.cat([box, torch.ones_like(sample['cls'][idx]), sample['cls'][idx]], dim=1).to(device=x.device))
        else:
            boxlist = self.yolo.get_results(pred, conf=conf)

        return self.cut_features(boxlist, feats, x.shape[2:]), boxlist

    def cut_features(self, boxlist, feat, oshape):
        batch = []
        for i, boxes in enumerate(boxlist):
            obox = []
            for box in boxes:
                conf, clas = box[4].item(), int(box[5])
                fshape = feat.shape[2:]
                scale = torch.tensor([fshape[0]/oshape[0], fshape[1]/oshape[1], fshape[0]/oshape[0], fshape[1]/oshape[1]], device=feat.device)
                coords = box[:4]*scale
                coords[[0,1]], coords[[2,3]] = torch.floor(coords[[0,1]]), torch.ceil(coords[[2,3]])
                coords = torch.clamp_min(coords.int(), 0)
                ofeat = feat[i,:,coords[1]:coords[3]+1, coords[0]:coords[2]+1].mean([1,2])
                obox.append([ofeat, conf, clas])
            batch.append(obox)
        return batch

# See AuXFT codebase for more details
class DemoVecs(nn.Module):
    def __init__(self, yolo, use_gt_boxes=False):
        super().__init__()

        self.use_gt_boxes = use_gt_boxes

        self.yolo = yolo
        self.yolo.eval()

        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dino.eval()

    def forward(self, x, conf=.3, sample=None):
        _, _, H, W = x.shape
        H1, W1 = H//14, W//14

        (pred, _), _ = self.yolo(x)

        if self.use_gt_boxes and sample is not None:
            boxlist = []
            for i in range(x.shape[0]):
                H1, W1 = sample['img'][i].shape[1:]
                idx = sample['batch_idx'] == i
                box = xywh2xyxy(sample['bboxes'][idx])*torch.tensor((W1,H1,W1,H1))
                boxlist.append(torch.cat([box, torch.ones_like(sample['cls'][idx]), sample['cls'][idx]], dim=1).to(device=x.device))
        else:
            boxlist = self.yolo.get_results(pred, conf=conf)

        return self.cut_images(boxlist, x)

    def cut_images(self, boxlist, x):
        batch = []
        for i, boxes in enumerate(boxlist):
            obox = []
            for box in boxes:
                conf, clas = box[4].item(), int(box[5])
                coords = box[:4]
                coords[[0,1]], coords[[2,3]] = torch.floor(coords[[0,1]]), torch.ceil(coords[[2,3]])
                coords = torch.clamp_min(coords.int(), 0)
                ix = F.interpolate(x[i:i+1,:,coords[1]:coords[3]+1, coords[0]:coords[2]+1], (224,224))
                ofeat = self.dino(ix, is_training=True)["x_norm_patchtokens"][0].mean(dim=0)
                obox.append([ofeat, conf, clas])
            batch.append(obox)
        return batch, boxlist

# LLaVA Oracle, produces h_i embeddings based on b_i
# (Paper Methodology - Multimodal Supervision Section)
class LlavaVecs(nn.Module):
    def __init__(self,
                 yolo,
                 use_gt_boxes=False):
        super().__init__()
        self.use_gt_boxes = use_gt_boxes

        self.yolo = yolo
        self.yolo.eval()

        print(torch.cuda.memory_allocated(), "/", torch.cuda.memory_reserved())

        self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf",
                                                            torch_dtype=torch.float16,
                                                            low_cpu_mem_usage=True,
                                                            ).to('cuda')
        print(torch.cuda.memory_allocated(), "/", torch.cuda.memory_reserved())


        self.model.eval()
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.prompt = "Describe accurately the visual appearance of %s using as many adjectives as possible. Include materials, colors, shapes, breeds, and any other relevant details.\n<image>\n"

    def fw_llava(self, x, y, y_fine=False, return_sentence=False, sample=False):
        x = x * 255

        prompt = self.prompt % y
        inputs = self.processor(images=x, text=prompt, return_tensors='pt').to(0, torch.float16)

        output = self.model.generate(**inputs,
                            max_new_tokens=300,
                            do_sample=True,
                            output_hidden_states=True,
                            return_dict_in_generate=True,
                            num_beams=3,             # Use beam search to enhance coherence
                            no_repeat_ngram_size=3)  # Avoid repetitive phrases
        out_sentence = self.processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
        states = torch.stack([el[-1] for el in output.hidden_states[1:-1]])
        text_features = states.mean(dim=[0,1,2])

        if return_sentence:
            return text_features, out_sentence
        return text_features

    def forward(self, x, conf=.3, sample=None):
        assert sample is not None, "Sample must be not None"

        _, _, H, W = x.shape
        H1, W1 = H//14, W//14

        (pred, _), _ = self.yolo(x)

        # create boxlists to cut the image either from gt or yolo pred
        if self.use_gt_boxes:
            boxlist = []
            for i in range(x.shape[0]):
                H1, W1 = sample['img'][i].shape[1:]
                idx = sample['batch_idx'] == i
                box = xywh2xyxy(sample['bboxes'][idx])*torch.tensor((W1,H1,W1,H1))
                boxlist.append(torch.cat([box, torch.ones_like(sample['cls'][idx]), sample['cls'][idx]], dim=1).to(device=x.device))
        else:
            boxlist = self.yolo.get_results(pred, conf=conf)

        return self.cut_images(boxlist, x, sample["names"])

    def cut_images(self, boxlist, x, names, fine_names=None):
        batch = []
        for i, boxes in enumerate(boxlist):
            obox = []
            for box in boxes:
                conf, clas = box[4].item(), int(box[5])
                coords = box[:4]
                coords[[0,1]], coords[[2,3]] = torch.floor(coords[[0,1]]), torch.ceil(coords[[2,3]])
                coords = torch.clamp_min(coords.int(), 0)
                ix = F.interpolate(x[i:i+1,:,coords[1]:coords[3]+1, coords[0]:coords[2]+1], (224,224))
                if fine_names:
                    ofeat = self.fw_llava(ix, names[clas], y_fine=fine_names[clas])
                else:
                    ofeat = self.fw_llava(ix, names[clas])
                obox.append([ofeat, conf, clas])
            batch.append(obox)
        return batch, boxlist

# CLIP Oracle, produces z_{V,i} embeddings based on b_i
# (Paper Methodology - Multimodal Supervision Section)
class ClipVecs(nn.Module):
    def __init__(self, yolo, use_gt_boxes=False, f_type = "spatial"):
        super().__init__()

        self.use_gt_boxes = use_gt_boxes

        # YOLO for object detection
        self.yolo = yolo
        self.yolo.eval()

        self.f_type = f_type

        # Load CLIP model
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(torch.float16)
        self.model.eval()
        self.model.to("cuda")

        # Prompt template for text descriptions
        self.prompt = "Describe accurately the visual appearance of %s using as many adjectives as possible. Include materials, colors, shapes, breeds, and any other relevant details.\n<image>\n"

    def fw_clip(self, x, y, names):
        x = x * 255
        x = x.to('cuda')

        # Preprocess inputs
        prompt = self.prompt % y
        inputs = self.processor(text=prompt, images=x, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")

        # Get CLIP embeddings
        with torch.no_grad():
            outputs = self.model(**inputs,
                                 output_hidden_states=False)
            cls_token = outputs.image_embeds[0] # CLS token

        return cls_token

    def forward(self, x, conf=.3, sample=None):
        assert sample is not None, "Sample must be not None"

        _, _, H, W = x.shape
        H1, W1 = H // 14, W // 14

        pred, _ = self.yolo(x)

        # Create boxlists to cut the image either from gt or yolo pred
        if self.use_gt_boxes:
            boxlist = []
            for i in range(x.shape[0]):
                H1, W1 = sample['img'][i].shape[1:]
                idx = sample['batch_idx'] == i
                box = xywh2xyxy(sample['bboxes'][idx]) * torch.tensor((W1, H1, W1, H1))
                boxlist.append(torch.cat([box, torch.ones_like(sample['cls'][idx]), sample['cls'][idx]], dim=1).to(device=x.device))
        else:
            boxlist = self.yolo.get_results(pred, conf=conf)

        return self.cut_images(boxlist, x, sample["names"])

    def cut_images(self, boxlist, x, names):
        batch = []
        for i, boxes in enumerate(boxlist):
            obox = []
            for box in boxes:
                conf, clas = box[4].item(), int(box[5])
                coords = box[:4]
                coords[[0, 1]], coords[[2, 3]] = torch.floor(coords[[0, 1]]), torch.ceil(coords[[2, 3]])
                coords = torch.clamp_min(coords.int(), 0)
                ix = F.interpolate(x[i:i+1, :, coords[1]:coords[3]+1, coords[0]:coords[2]+1], (224, 224))
                ofeat = self.fw_clip(ix, names[clas], names)
                obox.append([ofeat, conf, clas])
            batch.append(obox)
        return batch, boxlist

# Automatically merges and scales embeddings to produce u_i'
# (Paper Methodology - Dimensionality Reduction Section)
class LlavaClipVecs(nn.Module):
    def __init__(self, yolo, use_gt_boxes=False, eps=1e-5):
        super().__init__()

        self.use_gt_boxes = use_gt_boxes

        self.llava = LlavaVecs(yolo, use_gt_boxes)
        self.llava.eval()
        self.clip = ClipVecs(yolo, use_gt_boxes)
        self.clip.eval()

        self.eps = eps

    @torch.no_grad()
    def normalize_pca_vec(self, vec, a=18., b=.47, c=-.26):
        x = torch.arange(vec.shape[0], device=vec.device)
        std = a/((x+1)**b) + c
        return vec / (std + self.eps)

    def forward(self, x, conf=.3, sample=None):
        llava_vecs, llava_pred = self.llava(x, conf, sample)
        clip_vecs, clip_pred = self.clip(x, conf, sample)

        assert all(cl == cc for llava_box, clip_box \
                    in zip(llava_vecs, clip_vecs) for (_, _, cl), (_, _, cc) in \
                        zip(llava_box, clip_box)), f'embeddings misaligned'
        merged_vecs = [
            [
                (torch.cat([llava, clip * llava.norm()], dim=0), 1., c) \
                    for (llava, _, c), (clip, _, _) in zip(llava_box, clip_box)
            ] for llava_box, clip_box in zip(llava_vecs, clip_vecs)
        ]

        merged_pred = [(lp + cp) / 2 for lp, cp in zip(llava_pred, clip_pred)]

        return merged_vecs, merged_pred

# MOCHA translator architecture (t_S)
# (Paper Methodology - Feature Distillation Section)
class MochaFeatMap(nn.Module):
    def __init__(self, inchs=448, outchs=512, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.mean = nn.Parameter(torch.zeros((1,inchs)))
        self.std = nn.Parameter(torch.ones((1,inchs)))

        self.attn = nn.MultiheadAttention(inchs, 14, batch_first=True)
        self.bnattn = nn.LayerNorm(inchs)

        self.ffn1 = nn.Linear(inchs, 4*inchs)
        self.relu = nn.ReLU()
        self.ffn2 = nn.Linear(4*inchs, inchs)
        self.bnffn = nn.LayerNorm(inchs)

        self.out = nn.Linear(inchs, outchs, bias=False)

    @torch.no_grad()
    def normalize_pca_vec(self, vec, a=18., b=.47, c=-.26):
        x = torch.arange(vec.shape[0], device=vec.device)
        std = a/((x+1)**b) + c
        return vec / (std + self.eps)

    def forward(self, x):
        x = x.unsqueeze(0)

        # normalize the input data
        x = x - self.mean
        self.std.data.clamp_min_(self.eps)
        x0 = x / self.std

        x, _ = self.attn(x0, x0, x0, need_weights=False) # second output is useless
        x0 = x0 + self.bnattn(x)

        x = self.ffn1(x0)
        x = self.relu(x)
        x = self.ffn2(x)
        x = x0 + self.bnffn(x)

        x = self.out(x)
        return x[0]

# Simpler version of t_S, without attention or FFN
class SimpleFeatMap(nn.Module):
    def __init__(self, inchs=448, outchs=512, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.mean = nn.Parameter(torch.zeros((1,inchs)))
        self.std = nn.Parameter(torch.ones((1,inchs)))

        self.map = nn.Linear(inchs, outchs, bias=False)

    def forward(self, x):
        x = x.unsqueeze(0)

        # normalize the input data
        x = x - self.mean
        self.std.data.clamp_min_(self.eps)
        x = x / self.std

        x = self.map(x)
        return x[0]

# Extension of MochaFeatMap to compress the multi-scale
# features into the vectors f_{A,i}
# (Paper Methodology - Student Detector and Feature Aggregation Section)
class YoloMocha(nn.Module):
    def __init__(self,
                 yolo,
                 inchs=448,
                 outchs=256):
        super().__init__()

        self.yolo = yolo
        self.fmap = MochaFeatMap(inchs, outchs)

        self.use_gt_boxes = self.yolo.use_gt_boxes
        self.concatenate_chs = self.yolo.concatenate_chs
        self.pool_mode = self.yolo.pool_mode

    def forward(self, x, conf=.3, sample=None):
        vecs, pred = self.yolo(x, conf, sample)
        mapped = [[(self.fmap(v), p, c) for (v, p, c) in box] for box in vecs]
        return mapped, pred

# Same as above, but with the simpler translator
class YoloMap(nn.Module):
    def __init__(self,
                 yolo,
                 inchs=448,
                 outchs=256):
        super().__init__()

        self.yolo = yolo
        self.fmap = SimpleFeatMap(inchs, outchs)

        self.use_gt_boxes = self.yolo.use_gt_boxes
        self.concatenate_chs = self.yolo.concatenate_chs
        self.pool_mode = self.yolo.pool_mode

    def forward(self, x, conf=.3, sample=None):
        vecs, pred = self.yolo(x, conf, sample)
        mapped = [[(self.fmap(v), p, c) for (v, p, c) in box] for box in vecs]
        return mapped, pred
