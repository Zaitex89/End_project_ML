"""
Baselines to compare the multimodal network against.

    python baselines.py

A single deep model on its own proves nothing - the question an interviewer
asks is "compared to what?". This script answers it on the exact same grouped
split, with the same metrics:

  0. Dummy (always predict the majority class)     - the floor
  1. Logistic regression on metadata only
  2. Random forest on metadata only
  3. XGBoost on metadata only
  4. Logistic regression on frozen ImageNet embeddings (image only)
  5. XGBoost on ImageNet embeddings + metadata      - "classical" multimodal
  6. The fine-tuned multimodal CNN                  - read from evaluate.py

Steps 4-6 use PCA to squeeze the 2048-dim ResNet50 embeddings down to 128
components, which is where the data-reduction requirement earns its keep: the
classical models train in seconds instead of minutes and stop overfitting the
long tail of near-empty dimensions.

Writes artifacts/baseline_results.csv and artifacts/figures/baseline_comparison.png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

import config
from dataset import (HAM10000Dataset, build_splits, build_tabular_preprocessor,
                     eval_transform)

N_PCA = 128
EMB_CACHE = config.ARTIFACTS_DIR / "imagenet_embeddings.npz"


# ---------------------------------------------------------------- embeddings
@torch.no_grad()
def imagenet_embeddings(df: pd.DataFrame, tag: str) -> np.ndarray:
    """
    2048-dim features from an ImageNet-pretrained ResNet50 that has never seen
    HAM10000. Deliberately NOT the fine-tuned model: that one was trained on
    the training labels, so using it here would leak into the baselines and
    make the comparison meaningless.
    """
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model = model.to(config.DEVICE).eval()

    dummy_tab = np.zeros((len(df), 1), dtype=np.float32)
    ds = HAM10000Dataset(df, dummy_tab, eval_transform())
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)

    out = []
    for images, _tab, _y in tqdm(loader, desc=f"embeddings [{tag}]", leave=False):
        feats = model(images.to(config.DEVICE, non_blocking=True))
        out.append(feats.float().cpu().numpy())
    return np.concatenate(out)


def get_embeddings(train_df, val_df, test_df):
    """Compute once, cache to disk - this is the slow part of the script."""
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE)
        if all(len(z[k]) == len(d) for k, d in
               [("train", train_df), ("val", val_df), ("test", test_df)]):
            print(f"[info] Reusing cached embeddings: {EMB_CACHE.name}")
            return z["train"], z["val"], z["test"]
        print("[info] Cached embeddings do not match the split - recomputing.")

    tr = imagenet_embeddings(train_df, "train")
    va = imagenet_embeddings(val_df, "val")
    te = imagenet_embeddings(test_df, "test")
    np.savez_compressed(EMB_CACHE, train=tr, val=va, test=te)
    return tr, va, te


# ------------------------------------------------------------------- scoring
def score(name, y_true, y_pred, probs, n_features):
    classes = list(range(len(config.DX_CLASSES)))
    y_bin = label_binarize(y_true, classes=classes)
    return {
        "model": name,
        "features": n_features,
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_auc": float(roc_auc_score(y_bin, probs, average="macro",
                                         multi_class="ovr")),
        "macro_ap": float(average_precision_score(y_bin, probs, average="macro")),
    }


def fit_and_score(name, clf, X_tr, y_tr, X_te, y_te, weighted=True):
    """Fit on train, score on test. Rare classes are up-weighted, as in the CNN."""
    kwargs = {}
    if weighted and "class_weight" not in clf.get_params():
        kwargs["sample_weight"] = compute_sample_weight("balanced", y_tr)
    clf.fit(X_tr, y_tr, **kwargs)
    probs = clf.predict_proba(X_te)
    preds = probs.argmax(1)
    row = score(name, y_te, preds, probs, X_tr.shape[1])
    print(f"  {name:<46} macro-F1 {row['macro_f1']:.3f}  "
          f"bal-acc {row['balanced_accuracy']:.3f}  AUC {row['macro_auc']:.3f}")
    return row


def main():
    train_df, val_df, test_df = build_splits()
    y_tr = train_df["label"].values
    y_te = test_df["label"].values

    # Tabular features - same preprocessing as the neural network uses
    pre = build_tabular_preprocessor()
    T_tr = pre.fit_transform(train_df)
    T_te = pre.transform(test_df)
    print(f"[info] Tabular features: {T_tr.shape[1]}")

    results = []

    print("\n--- 0. Floor ---")
    results.append(fit_and_score(
        "0. Dummy (majority class)",
        DummyClassifier(strategy="most_frequent"),
        T_tr, y_tr, T_te, y_te, weighted=False))

    print("\n--- Metadata only (age, sex, localization) ---")
    results.append(fit_and_score(
        "1. Logistic regression (metadata)",
        LogisticRegression(max_iter=2000, class_weight="balanced",
                           random_state=config.SEED),
        T_tr, y_tr, T_te, y_te))
    results.append(fit_and_score(
        "2. Random forest (metadata)",
        RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                               class_weight="balanced_subsample",
                               n_jobs=-1, random_state=config.SEED),
        T_tr, y_tr, T_te, y_te))

    from xgboost import XGBClassifier
    xgb_params = dict(n_estimators=400, max_depth=6, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                      objective="multi:softprob",
                      num_class=len(config.DX_CLASSES),
                      random_state=config.SEED, n_jobs=-1)
    results.append(fit_and_score(
        "3. XGBoost (metadata)", XGBClassifier(**xgb_params),
        T_tr, y_tr, T_te, y_te))

    print("\n--- Image only: frozen ImageNet ResNet50 + PCA ---")
    E_tr, _E_va, E_te = get_embeddings(train_df, val_df, test_df)
    pca = PCA(n_components=N_PCA, random_state=config.SEED)
    P_tr = pca.fit_transform(E_tr)
    P_te = pca.transform(E_te)
    print(f"[info] PCA {E_tr.shape[1]} -> {N_PCA} dims, "
          f"{pca.explained_variance_ratio_.sum():.1%} of the variance kept")

    results.append(fit_and_score(
        "4. Logistic regression (ImageNet emb. + PCA)",
        LogisticRegression(max_iter=3000, class_weight="balanced",
                           random_state=config.SEED),
        P_tr, y_tr, P_te, y_te))

    print("\n--- Classical multimodal: embeddings + metadata ---")
    M_tr = np.hstack([P_tr, T_tr])
    M_te = np.hstack([P_te, T_te])
    results.append(fit_and_score(
        "5. XGBoost (ImageNet emb. + PCA + metadata)",
        XGBClassifier(**xgb_params), M_tr, y_tr, M_te, y_te))

    # 6. The fine-tuned network, if evaluate.py has been run
    if config.METRICS_PATH.exists():
        m = json.loads(config.METRICS_PATH.read_text())
        m = {**m, "model": "6. Fine-tuned multimodal CNN (fusion)",
             "features": "images + metadata"}
        m.pop("n_test", None)
        results.append(m)
        print(f"\n  {m['model']:<46} macro-F1 {m['macro_f1']:.3f}  "
              f"bal-acc {m['balanced_accuracy']:.3f}  AUC {m['macro_auc']:.3f}")
    else:
        print("\n[warning] No metrics_multimodal.json - run train.py + "
              "evaluate.py to get the deep model into the table.")

    # ------------------------------------------------------------- report
    res = pd.DataFrame(results)
    cols = ["model", "features", "accuracy", "balanced_accuracy", "macro_f1",
            "weighted_f1", "macro_auc", "macro_ap"]
    res = res[cols]
    res.to_csv(config.BASELINE_PATH, index=False)

    print("\n" + "=" * 100)
    print("BASELINE COMPARISON (test set, grouped on lesion_id)")
    print("=" * 100)
    print(res.round(3).to_string(index=False))
    print(f"\nSaved: {config.BASELINE_PATH}")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(res))
    ax.bar(x - 0.2, res["macro_f1"], 0.4, label="Macro-F1", color="steelblue")
    ax.bar(x + 0.2, res["balanced_accuracy"], 0.4, label="Balanced accuracy",
           color="indianred")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(". ", 1)[-1] for m in res["model"]],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on the same grouped test split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "baseline_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Figure: {config.FIGURES_DIR / 'baseline_comparison.png'}")


if __name__ == "__main__":
    main()
