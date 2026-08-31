"""
Multimodal fusionsmodel.

  Still ResNet50 (without the last layer) -> 2048-dim vektor
  Tabular branch: smal MLP -> 64-dim vector
  concatenation -> dense -> 7 klasser

This is the core of the project: structured + unstructured data in the same network.
"""
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

import config


class MultimodalNet(nn.Module):
    def __init__(self, tab_dim: int, n_classes: int = len(config.DX_CLASSES)):
        super().__init__()

        # Images branch: Pre trained ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V2
        backbone = resnet50(weights=weights)
        self.img_feat_dim = backbone.fc.in_features # 2048
        backbone.fc = nn.Identity() # remove clasification layer
        self.backbone = backbone

        # Tabular branch
        self.tabular = nn.Sequential(
            nn.Linear(tab_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(self.img_feat_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def freeze_backbone(self, freeze: bool = True):
        """Freeze/thaw CNN-backbone"""
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def forward_image_features(self, image):
        """Just image-embeddings (2048-dim) - used by extract_embeddings.py."""
        return self.backbone(image)

    def forward(self, image, tabular):
        img_f = self.backbone(image) # (B, 2048)
        tab_f = self.tabular(tabular) # (B, 64)
        fused = torch.cat([img_f, tab_f], dim=1)
        return self.head(fused) # (B, 7)


def build_model(tab_dim: int) -> "MultimodalNet":
    model = MultimodalNet(tab_dim=tab_dim)
    return model.to(config.DEVICE)
