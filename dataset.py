"""
Data prep + multimodal Dataset-class for HAM10000.

Here we do missing values, normalising, and the train/val/test-split.

The split is GROUPED on lesion_id and stratified on the diagnosis. HAM10000
contains up to 6 photos of the same lesion, so a plain row-wise split puts the
same lesion in both train and test - the model then recognises the lesion
instead of the diagnosis and every metric comes out too high.
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


# 4. Grouped + stratified split
def build_splits(df: pd.DataFrame | None = None, verbose: bool = True):
    """
    Split into train/val/test so that:
      - no lesion_id appears in more than one split (no leakage)
      - the class distribution is preserved in each split (stratified)

    StratifiedGroupKFold does both at once. We take one fold as test, one as
    val, and the rest as train. Same seed everywhere, so train.py, evaluate.py,
    baselines.py and eda.py all see the exact same split.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    if df is None:
        df = load_metadata()

    n_splits = max(3, round(1.0 / config.TEST_SPLIT))  # 0.15 -> 7 folds
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=config.SEED)
    folds = list(sgkf.split(df, df["label"], groups=df[config.GROUP_COL]))

    test_idx = folds[0][1]
    val_idx = folds[1][1]
    holdout = set(test_idx) | set(val_idx)
    train_idx = np.array([i for i in range(len(df)) if i not in holdout])

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    if verbose:
        overlap = (set(train_df[config.GROUP_COL]) & set(test_df[config.GROUP_COL])) | \
                  (set(train_df[config.GROUP_COL]) & set(val_df[config.GROUP_COL])) | \
                  (set(val_df[config.GROUP_COL]) & set(test_df[config.GROUP_COL]))
        assert not overlap, f"lesion_id leaked across splits: {len(overlap)}"
        n = len(df)
        print(f"[info] Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)} "
              f"({len(train_df)/n:.0%}/{len(val_df)/n:.0%}/{len(test_df)/n:.0%})")
        print(f"[info] lesion_id overlap between splits: 0 (grouped split OK)")

    return train_df, val_df, test_df


# 5. Build dataloaders (grouped split + unbalance handling)
def build_dataloaders():
    train_df, val_df, test_df = build_splits()

    pre = build_tabular_preprocessor()
    X_train = pre.fit_transform(train_df)
    X_val = pre.transform(val_df)
    X_test = pre.transform(test_df)

    joblib.dump(pre, config.PREPROCESSOR_PATH)
    tab_dim = X_train.shape[1]
    print(f"[info] Tabular features: {tab_dim} dimensions")

    train_ds = HAM10000Dataset(train_df, X_train, train_transform())
    val_ds = HAM10000Dataset(val_df, X_val, eval_transform())
    test_ds = HAM10000Dataset(test_df, X_test, eval_transform())

    class_counts = train_df["label"].value_counts().reindex(
        range(len(config.DX_CLASSES)), fill_value=0).values

    # Unbalance handling - exactly ONE strategy, chosen in config.py.
    # Sampler and class-weighted loss both push the rare classes up; doing both
    # double-counts the correction and wrecks precision on the majority class.
    common = dict(num_workers=config.NUM_WORKERS, pin_memory=True)
    if config.IMBALANCE_STRATEGY == "sampler":
        inv = 1.0 / np.maximum(class_counts, 1)
        sample_weights = inv[train_df["label"].values]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                                  sampler=sampler, **common)
    else:
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                                  shuffle=True, **common)
    print(f"[info] Imbalance strategy: {config.IMBALANCE_STRATEGY}")

    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                             shuffle=False, **common)

    # class_counts is also used for the weighted loss function in train.py
    return train_loader, val_loader, test_loader, tab_dim, class_counts


if __name__ == "__main__":
    # quick test
    tr, va, te, dim, counts = build_dataloaders()
    for imgs, tabs, labels in tr:
        print("Batch image:", imgs.shape, "tabular:", tabs.shape, "labels:", labels.shape)
        break