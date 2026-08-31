"""
Data prep + multimodal Dataset-class for HAM10000.

Here we do missing values,normalising, train/val/test-split.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

import config

# 1. Load and prep the metadata
def _build_image_index() -> dict:
    """Maps image_id."""
    index = {}
    for d in config.IMAGE_DIRS:
        if not d.exists():
            continue
        for p in d.glob("*.jpg"):
            index[p.stem] = p
    if not index:
        raise FileNotFoundError(
            "Found no .jpg-images. control check IMAGE_DIRS in config.py."
        )
    return index


def load_metadata() -> pd.DataFrame:
    """Reads CSV, removes rows without image."""
    df = pd.read_csv(config.METADATA_CSV)
    img_index = _build_image_index()
    df["path"] = df["image_id"].map(img_index)
    missing = df["path"].isna().sum()
    if missing:
        print(f"[info] {missing} row missing image and will be removed.")
    df = df.dropna(subset=["path"]).reset_index(drop=True)

    df["label"] = df["dx"].map({c: i for i, c in enumerate(config.DX_CLASSES)})
    return df


# columns for the tabular branch
NUMERIC_COLS = ["age"]
CATEGORICAL_COLS = ["sex", "localization"]


def build_tabular_preprocessor() -> ColumnTransformer:
    """
    - age: median imputation + standardization
    - sex / localization: imputation with 'unknown' + one-hot
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_COLS),
        ("cat", categorical, CATEGORICAL_COLS),
    ])


# 2. Transform the images
def train_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


# 3. Dataset-class
class HAM10000Dataset(Dataset):
    """Returns (images, tabular-vector, label) forevery row."""

    def __init__(self, df: pd.DataFrame, tabular_features: np.ndarray, transform):
        self.df = df.reset_index(drop=True)
        self.tabular = tabular_features.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        tab = torch.from_numpy(self.tabular[idx])
        label = int(row["label"])
        return image, tab, label


# 4. Build dataloaders (stratified split + unbalance handling)
def build_dataloaders():
    from sklearn.model_selection import train_test_split

    df = load_metadata()

    # Stratified split: train / (val + test), then val / test
    train_df, temp_df = train_test_split(
        df, test_size=config.VAL_SPLIT + config.TEST_SPLIT,
        stratify=df["label"], random_state=config.SEED,
    )
    rel_test = config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test,
        stratify=temp_df["label"], random_state=config.SEED,
    )

    pre = build_tabular_preprocessor()
    X_train = pre.fit_transform(train_df)
    X_val = pre.transform(val_df)
    X_test = pre.transform(test_df)

    joblib.dump(pre, config.PREPROCESSOR_PATH)
    tab_dim = X_train.shape[1]
    print(f"[info] Tabular features: {tab_dim} dimensions")
    print(f"[info] Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    train_ds = HAM10000Dataset(train_df, X_train, train_transform())
    val_ds = HAM10000Dataset(val_df, X_val, eval_transform())
    test_ds = HAM10000Dataset(test_df, X_test, eval_transform())

    # Unbalance handling: WeightedRandomSampler gives rare classes a higher chance
    class_counts = train_df["label"].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_df["label"].values]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    common = dict(num_workers=config.NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              sampler=sampler, **common)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                             shuffle=False, **common)

    # class_counts is also used for weighted lossfunction in train.py
    return train_loader, val_loader, test_loader, tab_dim, class_counts


if __name__ == "__main__":
    # quick test
    tr, va, te, dim, counts = build_dataloaders()
    for imgs, tabs, labels in tr:
        print("Batch image:", imgs.shape, "tabular:", tabs.shape, "labels:", labels.shape)
        break