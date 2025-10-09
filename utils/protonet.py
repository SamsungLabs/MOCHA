import torch

from models import YoloFeats, YoloVecs, DinoVecs, \
     DemoVecs, Conditional, BaseProtonet, SimpleShot, \
     LlavaVecs, ClipVecs, LlavaClipVecs, YoloMocha, YoloMap

# this helper function initializes the  
# correct protonet depending on
# on the command line arguments
# See AuXFT codebase for more details
def get_model_and_protonet(args, dataset):
    if args.pnet == 'cond':
        pnet = Conditional
    elif args.pnet == 'base':
        pnet = BaseProtonet
    elif args.pnet == 'simple':
        pnet = SimpleShot

    if args.model == 'base':
        yolo = YoloVecs(nc=31,
                        use_gt_boxes=True,
                        use_fcn=args.use_fcn,
                        is_base=True,
                        concatenate_chs=args.cat_chs,
                        mask_extra_coarse=args.mask_extra,
                        dataset=dataset,
                        pool_mode=args.pool_mode)
        odict = dict(yolo.state_dict()) # dict() needed to silence pylint bug
        for k, v in torch.load(args.base_ckpt, map_location='cpu').items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        yolo.load_state_dict(odict)
        yolo.to(args.device)
        yolo.eval()

        proto = pnet(consider_coarse=not args.coarse_disabled, # pylint: disable=all
                     norm=False)
        proto.to(args.device)

        return yolo, proto

    if args.model == 'residual':
        yolo = YoloVecs(nc=31,
                        use_gt_boxes=True,
                        use_fcn=args.use_fcn,
                        is_base=False,
                        mask_extra_coarse=args.mask_extra,
                        dataset=dataset,
                        pool_mode=args.pool_mode)
        odict = yolo.state_dict()
        for k, v in torch.load(args.ckpt, map_location='cpu').items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        yolo.load_state_dict(odict)
        yolo.to(args.device)
        yolo.eval()

        proto = pnet(consider_coarse=not args.coarse_disabled,
                     norm=False)
        proto.to(args.device)

        return yolo, proto

    if args.model == 'dino':
        yolo = YoloFeats(nc=31,
                         is_base=True)
        odict = yolo.state_dict()
        for k, v in torch.load(args.base_ckpt, map_location='cpu').items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        yolo.load_state_dict(odict)
        yolo.to(args.device)
        yolo.eval()

        dino = DinoVecs(yolo,
                        use_gt_boxes=True)
        dino.to(args.device)
        dino.eval()

        proto = pnet(consider_coarse=not args.coarse_disabled,
                     norm=False)
        proto.to(args.device)

        return dino, proto

    if args.model == 'demo':
        yolo = YoloFeats(nc=31,
                         is_base=True)
        odict = yolo.state_dict()
        for k, v in torch.load(args.base_ckpt, map_location='cpu').items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        yolo.load_state_dict(odict)
        yolo.to(args.device)
        yolo.eval()

        demo = DemoVecs(yolo,
                        use_gt_boxes=True)
        demo.to(args.device)
        demo.eval()

        proto = pnet(consider_coarse=not args.coarse_disabled,
                     norm=False)
        proto.to(args.device)

        return demo, proto

    if args.model == 'llava':
        yolo = YoloFeats(nc=31,
                         is_base=True)
        odict = yolo.state_dict()
        for k, v in torch.load(args.base_ckpt, map_location='cpu').items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        yolo.load_state_dict(odict)
        yolo.to(args.device)
        yolo.eval()

        demo = LlavaVecs(yolo,
                         use_gt_boxes=True)
        demo.to(args.device)
        demo.eval()

        proto = pnet(consider_coarse=not args.coarse_disabled,
                     norm=False)
        proto.to(args.device)

        return demo, proto

    if args.model == 'clip':
        yolo = YoloFeats(nc=31,
                         is_base=True)
        odict = yolo.state_dict()
        for k, v in torch.load(args.base_ckpt, map_location='cpu').items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        yolo.load_state_dict(odict)
        yolo.to(args.device)
        yolo.eval()

        demo = ClipVecs(yolo,
                        use_gt_boxes=True)
        demo.to(args.device)
        demo.eval()

        proto = pnet(consider_coarse=not args.coarse_disabled,
                     norm=False)
        proto.to(args.device)

        return demo, proto

    if args.model == 'llava-clip':
        yolo = YoloFeats(nc=31,
                         is_base=True)
        odict = yolo.state_dict()
        for k, v in torch.load(args.base_ckpt, map_location='cpu').items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        yolo.load_state_dict(odict)
        yolo.to(args.device)
        yolo.eval()

        demo = LlavaClipVecs(yolo,
                        use_gt_boxes=True)
        demo.to(args.device)
        demo.eval()

        proto = pnet(consider_coarse=not args.coarse_disabled,
                     norm=False)
        proto.to(args.device)

        return demo, proto

    if args.model == 'mocha':
        ckpt = torch.load(args.ckpt, map_location='cpu')
        if 'yolo' in ckpt and 'fmap' in ckpt:
            base_ckpt = ckpt['yolo']
            fmap_ckpt = ckpt['fmap']
        else:
            base_ckpt = torch.load(args.base_ckpt, map_location='cpu')
            fmap_ckpt = ckpt

        base = YoloVecs(nc=31,
                        use_gt_boxes=True,
                        use_fcn=False,
                        is_base=True,
                        concatenate_chs=True,
                        mask_extra_coarse=args.mask_extra,
                        dataset=dataset,
                        pool_mode=args.pool_mode)
        odict = dict(base.state_dict()) # dict() needed to silence pylint bug
        for k, v in base_ckpt.items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        base.load_state_dict(odict)
        base.to(args.device)
        base.eval()

        yolo = YoloMocha(base, outchs=args.pca_dim)
        odict = dict(yolo.fmap.state_dict()) # dict() needed to silence pylint bug
        for k, v in fmap_ckpt.items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        yolo.fmap.load_state_dict(odict)
        yolo.to(args.device)
        yolo.eval()

        proto = pnet(consider_coarse=not args.coarse_disabled, # pylint: disable=all
                     norm=False)
        proto.to(args.device)

        return yolo, proto

    if args.model in ['vild', 'ofa']:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        if 'yolo' in ckpt and 'fmap' in ckpt:
            base_ckpt = ckpt['yolo']
            fmap_ckpt = ckpt['fmap']
        else:
            base_ckpt = torch.load(args.base_ckpt, map_location='cpu')
            fmap_ckpt = ckpt

        base = YoloVecs(nc=31,
                        use_gt_boxes=True,
                        use_fcn=False,
                        is_base=True,
                        concatenate_chs=True,
                        mask_extra_coarse=args.mask_extra,
                        dataset=dataset,
                        pool_mode=args.pool_mode)
        odict = dict(base.state_dict()) # dict() needed to silence pylint bug
        for k, v in base_ckpt.items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        base.load_state_dict(odict)
        base.to(args.device)
        base.eval()

        yolo = YoloMap(base, outchs=args.pca_dim)
        odict = dict(yolo.fmap.state_dict()) # dict() needed to silence pylint bug
        for k, v in fmap_ckpt.items():
            if k.replace('module.', '') in odict:
                odict[k.replace('module.', '')] = v
            else:
                print('Ignoring key {%s}'%k.replace('module.', ''))
        yolo.fmap.load_state_dict(odict)
        yolo.to(args.device)
        yolo.eval()

        proto = pnet(consider_coarse=not args.coarse_disabled, # pylint: disable=all
                     norm=False)
        proto.to(args.device)

        return yolo, proto
