# Multimodal skin lesion classification (HAM10000)

Classifies 7 types of skin lesion from dermatoscopic images plus the patient
metadata that comes with them. The images go through a ResNet50, the metadata
(age, sex, body site) goes through a small MLP, and the two feature vectors are
concatenated before the classification head. So structured and unstructured
data end up in the same network.

Around that there is an EDA script with statistical tests, an unsupervised
clustering step, and six classical baselines to compare the network against.

I ran everything on an AMD RX 6800 XT with ROCm under Ubuntu. It will run on
NVIDIA/CUDA too since nothing is ROCm-specific except which PyTorch wheel you
install.

The presentation I gave on this is in the repo as
`multimodal_end_project.pptx`, 11 slides.

## Setup

PyTorch has to be installed separately from `requirements.txt`, otherwise pip
gives you the CPU or CUDA build instead of the ROCm one:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
pip install -r requirements.txt
```

Check that the GPU is actually visible:

```bash
python verify_setup.py
```

It prints the PyTorch version, the HIP version, the GPU name and the VRAM, and
does one matrix multiply on the device. If `torch.version.hip` is empty you
installed the wrong wheel.

## Data

HAM10000, from Kaggle or Harvard Dataverse. Unpack it like this:

```
data/
  HAM10000_metadata.csv
  HAM10000_images_part_1/    (5000 jpg)
  HAM10000_images_part_2/    (5015 jpg)
```

If you put it somewhere else, change `DATA_DIR` and `IMAGE_DIRS` at the top of
`config.py`. Everything else reads its paths from there.

10015 images, 7 classes, and one metadata row per image.

## Running it

In this order. Times are what it took on my machine (RX 6800 XT, 6 dataloader
workers).

| Step | Command | Time | Output |
|---|---|---|---|
| 1 | `python verify_setup.py` | instant | GPU check |
| 2 | `python eda.py` | ~45 s | `eda_report.txt` + 5 figures |
| 3 | `python dataset.py` | ~15 s | prints the split, sanity-checks a batch |
| 4 | `python extract_embeddings.py` | ~1 min | UMAP/cluster figure |
| 5 | `python train.py` | ~8 min | `best_model.pt`, `history.json`, training curve |
| 6 | `python evaluate.py` | ~25 s | report, metrics JSON, 3 figures |
| 7 | `python baselines.py` | ~1.5 min | `baseline_results.csv` + comparison figure |

Steps 2, 3 and 4 do not need a trained model, so you can run them first to see
what the data looks like. Step 7 reads `metrics_multimodal.json` if it exists,
which is how the network gets into the comparison table, so run 6 before 7.

Step 4 defaults to a plain ImageNet ResNet50 that has never seen a HAM10000
label. That is the version that is actually unsupervised. `--use-trained` runs
it on the fine-tuned backbone instead, which is a different question (what did
supervision do to the feature space) and not an unsupervised one.

Everything lands in `artifacts/`, figures in `artifacts/figures/`. The whole
directory is gitignored.

## Two things about the setup that matter

These are the two places where the obvious approach gives you numbers that are
too good, so they are worth knowing about before reading the results.

**The split is grouped on lesion_id.** HAM10000 has 10015 images but only 7470
unique lesions. 25% of the images are a second, third or sixth photo of a
lesion that is already in the data. A normal row-wise `train_test_split` puts
the same lesion on both sides, and then the model is being scored on lesions it
trained on. `build_splits()` in `dataset.py` uses `StratifiedGroupKFold` so a
lesion stays entirely in one split, and asserts that no lesion id crosses a
boundary. This costs a few points on every metric compared to the naive split,
which is the point.

**Only one imbalance correction is active at a time.** The classes are
imbalanced 58:1 (nv has 6705 images, df has 115). The first version used both a
`WeightedRandomSampler` and class weights in the loss, which corrects twice:
recall on the rare classes went up, but nv recall fell to 0.64 and melanoma
precision to 0.32. `config.IMBALANCE_STRATEGY` now picks one of
`loss_weights`, `sampler` or `none`, and `dataset.py`/`train.py` both read it so
the two can't be combined by accident. Default is `loss_weights`.

## Results

Test set, 1431 images, grouped split.

| Metric | Value |
|---|---|
| Accuracy | 0.816 |
| Balanced accuracy / macro recall | 0.682 |
| Macro F1 | 0.692 |
| Weighted F1 | 0.819 |
| Macro AUC-ROC (OvR) | 0.958 |
| Macro average precision | 0.731 |

Per class, from `artifacts/classification_report.txt`:

```
              precision    recall  f1-score   support
       akiec      0.413     0.565     0.477        46
         bcc      0.722     0.770     0.745        74
         bkl      0.606     0.618     0.612       157
          df      1.000     0.588     0.741        17
         mel      0.552     0.572     0.562       159
          nv      0.931     0.909     0.920       958
        vasc      0.833     0.750     0.789        20
```

df has a precision of 1.000 on 17 test images. That is 10 correct predictions
and no false positives, not a meaningful result. Same caveat for vasc.

The classes that actually limit the model are akiec (AP 0.46) and mel (AP 0.60).
39 real melanomas end up predicted as nv, which is the error that matters
clinically and is the obvious next thing to work on.

### Compared to simpler models

All on the same grouped test split, same metrics. From `baselines.py`.

| Model | Macro F1 | Balanced acc | Macro AUC |
|---|---|---|---|
| Dummy (always nv) | 0.115 | 0.143 | 0.500 |
| Logistic regression, metadata only | 0.216 | 0.390 | 0.784 |
| Random forest, metadata only | 0.232 | 0.315 | 0.749 |
| XGBoost, metadata only | 0.233 | 0.342 | 0.738 |
| Logistic regression on ImageNet embeddings + PCA | 0.463 | 0.551 | 0.883 |
| XGBoost on ImageNet embeddings + PCA + metadata | 0.467 | 0.420 | 0.924 |
| Fine-tuned multimodal network | 0.692 | 0.682 | 0.958 |

The embeddings for rows 5 and 6 come from an ImageNet ResNet50, not from the
fine-tuned model. Using the fine-tuned backbone there would leak the training
labels into the baseline and make the comparison meaningless.

Metadata on its own gets to about 0.23 macro F1. That matches what the EDA
found: age separates the classes clearly (Kruskal-Wallis H = 2271), but the
categorical variables are weak (Cramer's V 0.07 for sex, 0.22 for body site).
The images do the work, the metadata helps at the margin.

### Unsupervised part

ImageNet embeddings, PCA to 50 dims, then KMeans. Picking k by silhouette
without looking at the labels gives k=2, not 7. Adjusted Rand Index against the
true diagnoses is 0.02 even when you force k=7.

So the clusters do not line up with the diagnoses at all. The embeddings group
by colour and capture artefacts rather than by pathology. That is a real finding
and it is the argument for the supervised model: clustering alone gets you
nowhere on this dataset.

## Files

```
config.py               paths, hyperparameters, split and imbalance settings
verify_setup.py         GPU/ROCm check
eda.py                  EDA, statistical tests, figures, eda_report.txt
dataset.py              preprocessing, grouped split, Dataset class, dataloaders
model.py                the fusion network
train.py                training loop, saves best model on val macro F1
evaluate.py             confusion matrix, ROC, PR, classification report, metrics JSON
extract_embeddings.py   unsupervised: PCA, UMAP, KMeans, silhouette, ARI
baselines.py            the six comparison models

multimodal_end_project.pptx   the presentation, with speaker notes
```

## Config worth knowing about

Everything is in `config.py`.

- `IMBALANCE_STRATEGY` as described above.
- `GROUP_COL = "lesion_id"`, what the split groups on.
- `EPOCHS = 20`, `FREEZE_EPOCHS = 3`. The backbone is frozen for the first 3
  epochs so the new head can settle, then unfrozen with a low learning rate
  (`LR_BACKBONE = 1e-5` vs `LR_HEAD = 1e-3`).
- `BATCH_SIZE = 64` fits comfortably in the 16 GB on this card at 224x224
  with mixed precision. Drop it if you have less.
- `NUM_WORKERS = 6`. Set it to about the number of physical cores.

## What the preprocessing does

Age is the only column with missing values (57 rows, 0.57%), imputed with the
median and standardised. Sex and localization are imputed with the string
"unknown" and one-hot encoded.

That gives 18 features, not 19, which surprised me at first. The encoder is fit
on the training split only, and `acral` happens to be the one body site that
does not appear in the training fold. `handle_unknown="ignore"` deals with it
at test time. Fitting on all the data instead would fix the count and leak.

## Not done

- No Streamlit or Dash app. `evaluate.py` is the only interface.
- The EDA figures are matplotlib, not interactive Plotly.
- No threshold tuning. Everything is argmax over the softmax, which is not the
  right operating point for a melanoma screening tool.
- No Grad-CAM or any other look at what the network is attending to.
- Single train/test run, no cross-validation, so there are no error bars on any
  of the numbers above.
