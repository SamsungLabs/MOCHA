import torch
import os
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

if __name__ == "__main__":
    datasets = ["pod", "perseg", "icub", "core50"]
    splits = ["train"]

    X = []
    for dataset in datasets:
        for split in splits:
            merged_root = os.path.join("cache", "llava-clip", dataset, split)
            fnames = [fname for fname in sorted(os.listdir(merged_root)) if fname.endswith(".pth") and not fname.startswith(".")]
            for fname in tqdm(fnames, desc=f"Dataset: {dataset}, Split: {split}"):
                if '_pred' not in fname:
                    merged = torch.load(os.path.join(merged_root, fname), 'cpu', weights_only=True)
                    for x, _, _ in merged:
                        X.append(x)

    X = np.stack(X)
    pca = PCA(4608)
    pca.fit(X)

    pca_n = PCA(4608)
    X1 = X/np.linalg.norm(X, axis=1, keepdims=True)
    pca_n.fit(X1)

    components = pca.components_
    mean = pca.mean_
    np.savez("ckpts/pca_fs.npz", pca=components, mean=mean)

    Y = pca.transform(X)
    Y1 = (X - mean) @ components.T
    print("Shapes:", X.shape, Y.shape, Y1.shape, components.shape)
    print("Error sklearn/manual:", np.linalg.norm(Y-Y1))

    components = pca_n.components_
    mean = pca_n.mean_
    np.savez("ckpts/pca_fs_norm.npz", pca=components, mean=mean)

    Y = pca_n.transform(X1)
    Y1 = (X1 - mean) @ components.T
    print("Shapes:", X1.shape, Y.shape, Y1.shape, components.shape)
    print("Error sklearn/manual:", np.linalg.norm(Y-Y1))

    plt.plot(100*pca.explained_variance_ratio_, label="PCA")
    plt.plot(100*pca_n.explained_variance_ratio_, label="normed PCA")
    plt.yscale('log')
    plt.grid(True, 'both', 'both')
    plt.ylabel(r"Total % Explained Variance")
    plt.xlabel(r"Component #")
    plt.legend()
    plt.show()
