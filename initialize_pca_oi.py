import torch
import os
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import IncrementalPCA
from matplotlib import pyplot as plt

if __name__ == "__main__":
    datasets = ["openimages"]
    splits = ["train"]

    N = 4608 # same dimensionality as the vectors
    pca = IncrementalPCA(N)
    pca_n = IncrementalPCA(N)

    X = []
    for dataset in datasets:
        for split in splits:
            merged_root = os.path.join("cache", "llava-clip", dataset, split)
            fnames = [fname for fname in sorted(os.listdir(merged_root)) if fname.endswith(".pth") and not fname.startswith(".")]
            for fname in tqdm(fnames, desc=f"Dataset: {dataset}, Split: {split}", smoothing=0.):
                if '_pred' not in fname:
                    merged = torch.load(os.path.join(merged_root, fname), 'cpu', weights_only=True)
                    for x, _, _ in merged:
                        X.append(x)

                if len(X) >= 8*N: # learn pca in batches
                    X = np.stack(X)
                    pca.partial_fit(X)
                    pca_n.partial_fit(X/np.linalg.norm(X, axis=1, keepdims=True))
                    X = []

                    # save the current PCA
                    components = pca.components_
                    mean = pca.mean_
                    np.savez("ckpts/pca_oi.npz", pca=components, mean=mean)

                    # save the current normed PCA
                    components = pca_n.components_
                    mean = pca_n.mean_
                    np.savez("ckpts/pca_oi_norm.npz", pca=components, mean=mean)

    X = np.stack(X)
    if X.shape[0] >= N: # learn pca in batches
        pca.partial_fit(X)
        pca_n.partial_fit(X/np.linalg.norm(X, axis=1, keepdims=True))

        # save the current PCA
        components = pca.components_
        mean = pca.mean_
        np.savez("ckpts/pca_oi.npz", pca=components, mean=mean)

        # save the current normed PCA
        components = pca_n.components_
        mean = pca_n.mean_
        np.savez("ckpts/pca_oi_norm.npz", pca=components, mean=mean)
    else:
        print(f"Some vectors ({X.shape[0]}) could not be learned since {X.shape[0]} < {N}.")

    Y = pca.transform(X)
    Y1 = (X - pca.mean_) @ pca.components_.T
    print("Shapes:", X.shape, Y.shape, Y1.shape, pca.components_.shape)
    print("Error sklearn/manual:", np.linalg.norm(Y-Y1))

    X1 = X/np.linalg.norm(X, axis=1, keepdims=True)
    Y = pca_n.transform(X1)
    Y1 = (X1 - pca_n.mean_) @ pca_n.components_.T
    print("Shapes:", X1.shape, Y.shape, Y1.shape, pca_n.components_.shape)
    print("Error sklearn/manual:", np.linalg.norm(Y-Y1))

    plt.plot(100*pca.explained_variance_ratio_, label="PCA")
    plt.plot(100*pca_n.explained_variance_ratio_, label="normed PCA")
    plt.yscale('log')
    plt.grid(True, 'both', 'both')
    plt.ylabel(r"Total % Explained Variance")
    plt.xlabel(r"Component #")
    plt.legend()
    plt.show()
