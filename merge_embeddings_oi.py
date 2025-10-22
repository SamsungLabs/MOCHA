import torch
import os
from tqdm import tqdm

if __name__ == "__main__":
    datasets = ["openimages"]
    splits = ["train", "val"]

    for dataset in datasets:
        for split in splits:
            llava_root = os.path.join("cache", "llava", dataset, split)
            clip_root = os.path.join("cache", "clip", dataset, split)
            merged_root = os.path.join("cache", "llava-clip", dataset, split)
            os.makedirs(merged_root, exist_ok=True)

            fnames = [fname for fname in sorted(os.listdir(clip_root)) if fname.endswith(".pth") and not fname.startswith(".")]
            for fname in tqdm(fnames, desc=f"Dataset: {dataset}, Split: {split}"):
                if not os.path.isfile(os.path.join(merged_root, fname)):
                    if '_pred' in fname:
                        llava = torch.load(os.path.join(llava_root, fname), 'cpu', weights_only=True)
                        clip = torch.load(os.path.join(clip_root, fname), 'cpu', weights_only=True)
                        merged = (llava + clip) / 2.
                        torch.save(merged, os.path.join(merged_root, fname))
                    else:
                        llava_ = torch.load(os.path.join(llava_root, fname), 'cpu', weights_only=True)
                        if len(llava_) == 2 and len(llava_[1][0].shape) > 1:
                            llava = llava_[0][0]
                        else:
                            llava = llava_
                        clip = torch.load(os.path.join(clip_root, fname), 'cpu', weights_only=True)
                        assert all(lc == cc for (_, _, lc), (_, _, cc) in zip(llava, clip)), "Mismatched labels"
                        merged = [(torch.cat([lv, cv*lv.norm()], dim=0), 1., lc) for (lv, _, lc), (cv, _, _) in zip(llava, clip)] # clip has norm=1
                        torch.save(merged, os.path.join(merged_root, fname))
