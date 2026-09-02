"""
Unsupervised part: find structure in the data WITHOUT using the labels.

    python extract_embeddings.py                # honest version, ImageNet weights
    python extract_embeddings.py --use-trained  # what the fine-tuned model learned

By default this runs on a plain ImageNet-pretrained ResNet50 that has never
seen a HAM10000 label. That is the version you can actually call unsupervised:
it answers "if we had no diagnoses at all, would the images fall into groups?"
Running it on the fine-tuned model instead is a different, also useful question
- "what did supervision do to the feature space?" - but it is not unsupervised,
so it is behind a flag.

Steps:
  1. All images through the CNN backbone -> 2048-dim embeddings
  2. PCA 2048 -> 50 (denoise + speed), then UMAP -> 2D for plotting
  3. KMeans on the PCA space, k chosen by silhouette
  4. Only at the very end: compare the clusters to the true diagnoses (ARI)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

import config
from dataset import HAM10000Dataset, eval_transform, load_metadata

N_PCA = 50
K_RANGE = range(2, 11)


@torch.no_grad()
def compute_embeddings(model, loader):
    model.eval()
    embs, labels = [], []
    for images, _tabs, y in tqdm(loader, desc="embeddings", leave=False):
        feats = model(images.to(config.DEVICE, non_blocking=True))  # (B, 2048)
        embs.append(feats.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(embs), np.concatenate(labels)


def build_backbone(use_trained: bool):
    if use_trained and config.MODEL_PATH.exists():
        from model import build_model
        ckpt = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        full = build_model(ckpt["tab_dim"])
        full.load_state_dict(ckpt["state_dict"])
        print("[info] Using the fine-tuned backbone (NOT unsupervised).")
        return full.backbone.to(config.DEVICE), "finetuned"
    if use_trained:
        print("[warning] No trained model found - falling back to ImageNet.")
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    print("[info] Using ImageNet weights - no HAM10000 label has been seen.")
    return net.to(config.DEVICE), "imagenet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-trained", action="store_true",
                    help="use the fine-tuned backbone instead of ImageNet")
    args = ap.parse_args()

    df = load_metadata()
    dummy_tab = np.zeros((len(df), 1), dtype=np.float32)
    ds = HAM10000Dataset(df, dummy_tab, eval_transform())
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)

    backbone, tag = build_backbone(args.use_trained)
    emb, labels = compute_embeddings(backbone, loader)
    np.save(config.ARTIFACTS_DIR / f"embeddings_{tag}.npy", emb)
    print(f"[info] Embeddings: {emb.shape}")

    # Data reduction: PCA first. 2048 dims is mostly noise for clustering, and
    # UMAP on 50 dims is far faster than on 2048.
    pca = PCA(n_components=N_PCA, random_state=config.SEED)
    emb_pca = pca.fit_transform(emb)
    print(f"[info] PCA 2048 -> {N_PCA} dims, "
          f"{pca.explained_variance_ratio_.sum():.1%} of the variance kept")

    # Pick k WITHOUT looking at the labels - silhouette on a subsample
    idx = np.random.default_rng(config.SEED).choice(
        len(emb_pca), size=min(3000, len(emb_pca)), replace=False)
    sil = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10)
        sil[k] = silhouette_score(emb_pca[idx], km.fit_predict(emb_pca[idx]))
    best_k = max(sil, key=sil.get)
    print("[info] Silhouette per k: "
          + ", ".join(f"k={k}:{v:.3f}" for k, v in sil.items()))
    print(f"[info] Best k by silhouette: {best_k} "
          f"(the data has {len(config.DX_CLASSES)} true classes)")

    kmeans = KMeans(n_clusters=best_k, random_state=config.SEED, n_init=10)
    clusters = kmeans.fit_predict(emb_pca)

    # UMAP for the 2D picture
    try:
        import umap
        emb_2d = umap.UMAP(n_neighbors=15, min_dist=0.1,
                           random_state=config.SEED).fit_transform(emb_pca)
        method = "UMAP"
    except ImportError:
        print("[warning] umap-learn missing, falls back to PCA.")
        emb_2d = emb_pca[:, :2]
        method = "PCA"

    # Now, and only now, we bring in the labels to judge the clustering
    ari = adjusted_rand_score(labels, clusters)
    ari_7 = adjusted_rand_score(
        labels, KMeans(n_clusters=len(config.DX_CLASSES),
                       random_state=config.SEED, n_init=10).fit_predict(emb_pca))
    print(f"Adjusted Rand Index, k={best_k} (silhouette choice): {ari:.3f}")
    print(f"Adjusted Rand Index, k=7  (one cluster per diagnosis): {ari_7:.3f}")
    print("-> ARI near 0 means the unsupervised structure does NOT line up with")
    print("   the diagnoses. The visual grouping is driven by colour, hair and")
    print("   capture artefacts, not by pathology. That is the hypothesis this")
    print("   step generates, and it is exactly why supervised fine-tuning on")
    print("   labelled images is needed.")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for i, name in enumerate(config.DX_CLASSES):
        m = labels == i
        axes[0].scatter(emb_2d[m, 0], emb_2d[m, 1], s=5, label=name, alpha=0.6)
    axes[0].set_title(f"{method} - coloured by TRUE diagnosis")
    axes[0].legend(fontsize=8, markerscale=2)

    axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=clusters, cmap="tab10",
                    s=5, alpha=0.6)
    axes[1].set_title(f"{method} - KMeans k={best_k} (ARI {ari:.3f})")

    axes[2].plot(list(sil), [sil[k] for k in sil], "o-")
    axes[2].axvline(len(config.DX_CLASSES), color="grey", ls="--",
                    label="true number of classes")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Silhouette")
    axes[2].set_title("Choosing k without labels")
    axes[2].legend()

    fig.suptitle(f"Unsupervised structure - {tag} backbone")
    fig.tight_layout()
    out = config.FIGURES_DIR / f"umap_clusters_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
