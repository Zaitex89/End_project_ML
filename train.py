"""
Trains the multimodal model.

    python train.py

Uses:
  - mixed precision (torch.cuda.amp) -> quicker + less VRAM
  - class weight in the loss function -> handles the big unbalance
  - two phase transfer learning -> first frozen backbone, then fine tuning
saves the best model (after macro-F1 on the validation) to artifacts/.
"""
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

    # Class weight: rare classes gets bigger weight in the loss function
    weights = (class_counts.sum() / (len(class_counts) * class_counts))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.cuda.amp.GradScaler()
    best_f1 = 0.0

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

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({"state_dict": model.state_dict(), "tab_dim": tab_dim},
                       config.MODEL_PATH)
            print(f"    * new best model saved (macro-F1 {best_f1:.3f})")

    print(f"\nDone. Best val macro-F1: {best_f1:.3f}")
    print(f"Model saved: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
