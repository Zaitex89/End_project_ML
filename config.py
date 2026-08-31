"""
Central config for whole project
only change things in here, whole project reads from here.
"""
from pathlib import Path
import torch

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"                       # root map for dataset
METADATA_CSV = DATA_DIR / "HAM10000_metadata.csv"     # metadata-file

IMAGE_DIRS = [
    DATA_DIR / "HAM10000_images_part_1",
    DATA_DIR / "HAM10000_images_part_2",
]

ARTIFACTS_DIR = PROJECT_DIR / "artifacts" # models, figures, encoders
ARTIFACTS_DIR.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS_DIR / "best_model.pt"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"

# The 7 classes in HAM10000
DX_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DX_FULL_NAMES = {
    "akiec": "Actinic keratoses / intraepitelial carcinom",
    "bcc":   "Basalcellscancer",
    "bkl":   "Benign keratos",
    "df":    "Dermatofibrom",
    "mel":   "Melanom",
    "nv":    "Melanocytiskt nevus (vanlig mola)",
    "vasc":  "Vaskulara lesioner",
}

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 6
EPOCHS = 20
FREEZE_EPOCHS = 3
LR_HEAD = 1e-3
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# ImageNet-statistik
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
