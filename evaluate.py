"""
Evaluation of the trained model.

    python evaluate.py

Produces:
  - artifacts/classification_report.txt
  - artifacts/metrics_multimodal.json   (picked up by baselines.py)
  - artifacts/figures/confusion_matrix.png
  - artifacts/figures/roc_curves.png    (AUC-ROC, one-vs-rest per class)
  - artifacts/figures/pr_curves.png     (precision-recall per class)
Precision-recall is particulary important since the dataset is unbalanced.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

import config
from dataset import build_dataloaders
from model import build_model


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for images, tabs, labels in loader:
        images = images.to(config.DEVICE)
        tabs = tabs.to(config.DEVICE)
        logits = model(images, tabs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    _, _, test_loader, tab_dim, _ = build_dataloaders()

    ckpt = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model = build_model(ckpt["tab_dim"])
    model.load_state_dict(ckpt["state_dict"])

    probs, y_true = collect_predictions(model, test_loader)
    y_pred = probs.argmax(1)
    names = config.DX_CLASSES

    # 1. Classification report
    report = classification_report(y_true, y_pred, target_names=names, digits=3)
    print(report)
    (config.ARTIFACTS_DIR / "classification_report.txt").write_text(report)

    #  2. Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 3. AUC-ROC (one-vs-rest)
    y_bin = label_binarize(y_true, classes=list(range(len(names))))
    macro_auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    print(f"Macro AUC-ROC (OvR): {macro_auc:.3f}")

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, name in enumerate(names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        auc_i = roc_auc_score(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc_i:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC per class  |  macro AUC {macro_auc:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    # 4. Precision-recall per class
    macro_ap = average_precision_score(y_bin, probs, average="macro")
    print(f"Macro average precision (PR-AUC): {macro_ap:.3f}")
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, name in enumerate(names):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ax.plot(rec, prec, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall per class  |  macro AP {macro_ap:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "pr_curves.png", dpi=150)
    plt.close(fig)

    # 5. Headline numbers, so baselines.py can put the fusion model in the table
    metrics = {
        "model": "Multimodal CNN + tabular (fusion)",
        "per_class_auc": {n: float(roc_auc_score(y_bin[:, i], probs[:, i]))
                          for i, n in enumerate(names)},
        "per_class_ap": {n: float(average_precision_score(y_bin[:, i], probs[:, i]))
                         for i, n in enumerate(names)},
        "confusion_matrix": cm.tolist(),
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_auc": float(macro_auc),
        "macro_ap": float(macro_ap),
        "n_test": int(len(y_true)),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("\n" + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()
                            if isinstance(v, float)))
    print(f"Figures saved in {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
