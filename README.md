# Multimodal Skin Diagnostics (HAM10000)

Multimodal deep learning that combines **dermatoscopic images** (unstructured data)
with **patient metadata** (structured data) in a single neural network, to classify
7 skin lesions. Includes unsupervised analysis (embeddings + UMAP + clustering),
supervised deep learning, and full evaluation.

Built for **AMD RX 6800 XT + ROCm on Linux (Ubuntu)**.

## 1. Install PyTorch with ROCm

Install PyTorch SEPARATELY (not via requirements.txt) so you get the ROCm build:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
pip install -r requirements.txt
```

Confirm the GPU is detected:

```bash
python verify_setup.py
```

Should print "AMD Radeon RX 6800" and "Everything looks good".

## 2. Download the dataset

Get HAM10000 (Kaggle or Harvard Dataverse) and lay it out like this:

```
data/
  HAM10000_metadata.csv
  HAM10000_images_part_1/   (*.jpg)
  HAM10000_images_part_2/   (*.jpg)
```

Adjust the paths at the top of `config.py` if needed.

## 3. Run order

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `python verify_setup.py`        | Confirms ROCm + GPU |
| 2 | `python dataset.py`             | Quick test: distribution + batch shape |
| 3 | `python extract_embeddings.py`  | Unsupervised: UMAP + clustering |
| 4 | `python train.py`               | Trains the fusion model |
| 5 | `python evaluate.py`            | Confusion matrix, ROC, PR, report |

All outputs (model, figures, encoders) are written to `artifacts/`.

## Project structure

- `config.py` - all paths and hyperparameters in one place
- `dataset.py` - data preparation + multimodal Dataset class
- `model.py` - the fusion network (ResNet50 + tabular MLP)
- `train.py` - training loop (mixed precision, class weights, two-phase)
- `evaluate.py` - evaluation
- `extract_embeddings.py` - unsupervised analysis

## Next steps (bonus points)

- EDA notebook with interactive Plotly visualizations
- A baseline to compare against (e.g. XGBoost on metadata only)
- Streamlit app for a live demo (`app.py`)