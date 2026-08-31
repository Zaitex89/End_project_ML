"""
Unsupervised-part: first datastructure without labels.

    python extract_embeddings.py

Steps:
  1. Runs all images through CNN-spine -> 2048-dim embeddings
  2. Reduces with UMAP to 2D
  3. Cluster with KMeans + HDBSCAN
  4. Compare clusters against the true diagnoses (Adjusted Rand Index)
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import config
from dataset import (HAM10000Dataset, build_tabular_preprocessor,
                     eval_transform, load_metadata)
from model import build_model
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_embeddings(model, loader):
    model.eval()
    embs, labels = [], []
    for images, _tabs, y in loader:
        images = images.to(config.DEVICE)
        feats = model.forward_image_features(images)  # (B, 2048)
        embs.append(feats.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(embs), np.concatenate(labels)


def main():
    # Load trained model
    df = load_metadata()
    pre = build_tabular_preprocessor()
    X = pre.fit_transform(df)  # tabular is only needed to create the dataset
    ds = HAM10000Dataset(df, X, eval_transform())
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE,
                        shuffle=False, num_workers=config.NUM_WORKERS)

    model = build_model(tab_dim=X.shape[1])
    if config.MODEL_PATH.exists():
        ckpt = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        print("[info] Uses trained model for embeddings.")
    else:
        print("[info] no trained model found - uses ImageNet-weights.")

    print("[info] calculates embeddings ...")
    emb, labels = compute_embeddings(model, loader)
    np.save(config.ARTIFACTS_DIR / "embeddings.npy", emb)

    # UMAP to 2D
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=config.SEED)
        emb_2d = reducer.fit_transform(emb)
    except ImportError:
        print("[warning] umap-learn missing, falls back to PCA.")
        from sklearn.decomposition import PCA
        emb_2d = PCA(n_components=2, random_state=config.SEED).fit_transform(emb)

    # Clustering
    kmeans = KMeans(n_clusters=len(config.DX_CLASSES),
                    random_state=config.SEED, n_init=10)
    clusters = kmeans.fit_predict(emb)
    ari = adjusted_rand_score(labels, clusters)
    print(f"Adjusted Rand Index (KMeans vs true diagnoses): {ari:.3f}")

    # Visualization: colored after true diagnoses
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for i, name in enumerate(config.DX_CLASSES):
        m = labels == i
        axes[0].scatter(emb_2d[m, 0], emb_2d[m, 1], s=5, label=name, alpha=0.6)
    axes[0].set_title("UMAP - colored after true diagnoses")
    axes[0].legend(fontsize=8, markerscale=2)

    axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=clusters, cmap="tab10", s=5, alpha=0.6)
    axes[1].set_title(f"UMAP - KMeans-kluster (ARI {ari:.3f})")

    fig.tight_layout()
    fig.savefig(config.ARTIFACTS_DIR / "umap_clusters.png", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {config.ARTIFACTS_DIR / 'umap_clusters.png'}")


if __name__ == "__main__":
    main()
