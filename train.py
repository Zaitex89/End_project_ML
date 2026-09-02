"""
Trains the multimodal model.

    python train.py

Uses:
  - mixed precision (torch.amp) -> quicker + less VRAM
  - ONE imbalance strategy, picked in config.IMBALANCE_STRATEGY
  - two phase transfer learning -> first frozen backbone, then fine tuning
saves the best model (after macro-F1 on the validation) to artifacts/.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

import config
from dataset import build_dataloaders
from model import build_model


def run_epoch(model, loader, criterion, optimizer, scaler, train: bool):
    model.train() if train else model.eval()
    losses, all_preds, all_labels = [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, tabs, labels in tqdm(loader, leave=False):
            images = images.to(config.DEVICE, non_blocking=True)
            tabs = tabs.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            if train:
                optimizer.zero_grad()

            # Mixed precision
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images, tabs)
                loss = criterion(logits, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            losses.append(loss.item())
            all_preds.append(logits.argmax(1).detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labs = np.concatenate(all_labels)
    macro_f1 = f1_score(labs, preds, average="macro")
    return float(np.mean(losses)), macro_f1


def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    train_loader, val_loader, _, tab_dim, class_counts = build_dataloaders()
    model = build_model(tab_dim)

    # The loss is weighted only when the dataloader is NOT already resampling.
    # See the note in config.IMBALANCE_STRATEGY.
    if config.IMBALANCE_STRATEGY == "loss_weights":
        weights = class_counts.sum() / (len(class_counts) * class_counts)
        class_weights = torch.tensor(weights, dtype=torch.float32,
                                     device=config.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda")
    best_f1 = 0.0
    history = []

    # Phase 1: frozen backbone, only train new layers
    model.freeze_backbone(True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LR_HEAD, weight_decay=config.WEIGHT_DECAY,
    )

    for epoch in range(1, config.EPOCHS + 1):
        if epoch == config.FREEZE_EPOCHS + 1:
            # Phase 2: thaw the backbone, low learning rate
            print("--> Thaws the CNN-backbone (phase 2)")
            model.freeze_backbone(False)
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": config.LR_BACKBONE},
                {"params": model.tabular.parameters(), "lr": config.LR_HEAD},
                {"params": model.head.parameters(), "lr": config.LR_HEAD},
            ], weight_decay=config.WEIGHT_DECAY)

        tr_loss, tr_f1 = run_epoch(model, train_loader, criterion,
                                   optimizer, scaler, train=True)
        va_loss, va_f1 = run_epoch(model, val_loader, criterion,
                                   optimizer, scaler, train=False)

        print(f"Epoch {epoch:02d} | "
              f"train loss {tr_loss:.3f} f1 {tr_f1:.3f} | "
              f"val loss {va_loss:.3f} f1 {va_f1:.3f}")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_f1": tr_f1,
                        "val_loss": va_loss, "val_f1": va_f1})

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({"state_dict": model.state_dict(), "tab_dim": tab_dim},
                       config.MODEL_PATH)
            print(f"    * new best model saved (macro-F1 {best_f1:.3f})")

    (config.ARTIFACTS_DIR / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history)

    print(f"\nDone. Best val macro-F1: {best_f1:.3f}")
    print(f"Model saved: {config.MODEL_PATH}")


def plot_history(history):
    """Loss + macro-F1 per epoch - shows whether the fine-tuning overfits."""
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(ep, [h["train_f1"] for h in history], label="train")
    axes[1].plot(ep, [h["val_f1"] for h in history], label="val")
    axes[1].set_title("Macro-F1")
    for ax in axes:
        ax.axvline(config.FREEZE_EPOCHS + 0.5, color="grey", ls="--", alpha=0.6)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.suptitle("Training history (dashed line = backbone unfrozen)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "training_history.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
