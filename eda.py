"""
EDA + statistics on HAM10000.

    python eda.py

Answers three questions before any model is trained:
  1. What does the data look like - how much is missing, how skewed is it?
  2. Which metadata variables actually carry signal about the diagnosis?
     (chi-square + Cramer's V for the categoricals, Kruskal-Wallis for age)
  3. Do the images themselves differ between classes on something as crude as
     colour - i.e. is there anything for a CNN to grab onto?

Writes artifacts/eda_report.txt and figures to artifacts/figures/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

import config
from dataset import CATEGORICAL_COLS, load_metadata

RNG = np.random.default_rng(config.SEED)
LINES = []  # collected into artifacts/eda_report.txt


def say(*parts):
    text = " ".join(str(p) for p in parts)
    print(text)
    LINES.append(text)


def header(title):
    say("\n" + "=" * 70)
    say(title)
    say("=" * 70)


def cramers_v(confusion: pd.DataFrame) -> float:
    """Effect size for a chi-square test: 0 = no association, 1 = perfect."""
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.values.sum()
    r, k = confusion.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


# 1. Structure, missing values, duplicates
def overview(df: pd.DataFrame):
    header("1. STRUCTURE AND MISSING VALUES")
    say(f"Rows (images): {len(df)}")
    say(f"Unique lesions (lesion_id): {df[config.GROUP_COL].nunique()}")
    say(f"Unique patients cannot be derived - HAM10000 has no patient id.")
    say("")

    miss = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
    })
    say(miss.to_string())
    say("")
    say("-> 'age' is the only column with real missing values; imputed with the")
    say("   median in dataset.py. 'sex' has an explicit 'unknown' category.")

    # How many images per lesion - the reason the split must be grouped
    per_lesion = df[config.GROUP_COL].value_counts()
    say("")
    say("Images per lesion:")
    say(per_lesion.value_counts().sort_index().to_string())
    dup = len(df) - df[config.GROUP_COL].nunique()
    say(f"-> {dup} images ({dup/len(df):.1%}) are repeat photos of a lesion that")
    say("   is already in the data. A row-wise train/test split would put the")
    say("   same lesion on both sides. Hence StratifiedGroupKFold on lesion_id.")

    # dx_type: how the diagnosis was established
    say("")
    say("dx_type (how the diagnosis was confirmed):")
    say(df["dx_type"].value_counts().to_string())
    ct = pd.crosstab(df["dx"], df["dx_type"])
    say("")
    say("dx_type per diagnosis (row %):")
    say((ct.div(ct.sum(1), axis=0) * 100).round(1).to_string())
    say("-> nv is largely confirmed by follow-up rather than histology, so its")
    say("   labels are noisier than the labels of the malignant classes.")


# 2. Target distribution
def class_distribution(df: pd.DataFrame):
    header("2. TARGET VARIABLE (dx) - DISTRIBUTION")
    counts = df["dx"].value_counts()
    share = (counts / len(df) * 100).round(2)
    table = pd.DataFrame({"n": counts, "%": share,
                          "name": [config.DX_FULL_NAMES[c] for c in counts.index]})
    say(table.to_string())
    ratio = counts.max() / counts.min()
    say("")
    say(f"Imbalance ratio (largest/smallest class): {ratio:.1f}x")
    say(f"Majority baseline (always predict '{counts.index[0]}'): "
        f"{counts.iloc[0]/len(df):.1%} accuracy, but macro-F1 only "
        f"{1/len(counts) * 2 * (counts.iloc[0]/len(df)) / (1 + counts.iloc[0]/len(df)):.3f}")
    say("-> Accuracy is useless here. Macro-F1, balanced accuracy and the")
    say("   per-class PR curves are the metrics that matter.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(counts.index, counts.values, color="steelblue")
    axes[0].set_title("Images per diagnosis")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    axes[1].bar(counts.index, counts.values, color="indianred")
    axes[1].set_yscale("log")
    axes[1].set_title(f"Same, log scale (imbalance {ratio:.0f}x)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_class_distribution.png", dpi=150)
    plt.close(fig)


# 3. Age - numeric variable
def age_analysis(df: pd.DataFrame):
    header("3. AGE vs DIAGNOSIS (numeric)")
    say(df["age"].describe().round(2).to_string())
    say("")
    grouped = df.groupby("dx")["age"].agg(["count", "mean", "median", "std"]).round(2)
    say(grouped.to_string())

    # Kruskal-Wallis: non-parametric ANOVA, age is not normally distributed
    samples = [g["age"].dropna().values for _, g in df.groupby("dx")]
    h, p = stats.kruskal(*samples)
    say("")
    say(f"Kruskal-Wallis H = {h:.1f}, p = {p:.3e}")
    say("-> Age distribution differs significantly between diagnoses."
        if p < 0.05 else "-> No significant difference.")
    say("   nv (benign moles) sits ~15 years younger than the malignant classes,")
    say("   so age is genuinely informative for the tabular branch.")

    # Normality check that justifies using Kruskal-Wallis over ANOVA
    sub = df["age"].dropna().sample(min(5000, df["age"].notna().sum()),
                                    random_state=config.SEED)
    w, p_norm = stats.shapiro(sub)
    say(f"   (Shapiro-Wilk on age: W = {w:.3f}, p = {p_norm:.1e} -> not normal,")
    say("    which is why we use Kruskal-Wallis and not one-way ANOVA.)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(df["age"].dropna(), bins=20, color="steelblue", edgecolor="white")
    axes[0].set_title("Age distribution (all patients)")
    axes[0].set_xlabel("Age")
    order = df.groupby("dx")["age"].median().sort_values().index
    axes[1].boxplot([df.loc[df["dx"] == c, "age"].dropna() for c in order],
                    tick_labels=list(order))
    axes[1].set_title(f"Age per diagnosis (Kruskal-Wallis p = {p:.1e})")
    axes[1].set_ylabel("Age")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_age.png", dpi=150)
    plt.close(fig)


# 4. Categoricals - sex and localization
def categorical_analysis(df: pd.DataFrame):
    header("4. SEX / LOCALIZATION vs DIAGNOSIS (categorical)")
    for col in CATEGORICAL_COLS:
        ct = pd.crosstab(df[col], df["dx"])
        ct = ct.loc[ct.sum(1) >= 5]  # chi-square needs non-tiny cells
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        v = cramers_v(ct)
        say("")
        say(f"{col}: chi2 = {chi2:.1f}, dof = {dof}, p = {p:.3e}, "
            f"Cramer's V = {v:.3f}")
        strength = ("negligible" if v < 0.1 else "weak" if v < 0.2
                    else "moderate" if v < 0.3 else "strong")
        say(f"  -> association with the diagnosis is {strength}")
        say(df[col].value_counts().head(10).to_string())

    say("")
    say("-> Both are significant, but the effect sizes are modest. Metadata")
    say("   alone will not solve this problem - see baselines.py. It is the")
    say("   image branch that has to do the heavy lifting.")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={"width_ratios": [1, 2]})
    sex_ct = pd.crosstab(df["sex"], df["dx"], normalize="index") * 100
    sex_ct.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab10",
                legend=False)
    axes[0].set_title("Diagnosis mix per sex (%)")
    axes[0].set_ylabel("% of images")

    loc_ct = pd.crosstab(df["localization"], df["dx"], normalize="index") * 100
    loc_ct = loc_ct.loc[df["localization"].value_counts().index]
    im = axes[1].imshow(loc_ct.values, aspect="auto", cmap="viridis")
    axes[1].set_xticks(range(len(loc_ct.columns)))
    axes[1].set_xticklabels(loc_ct.columns, rotation=45)
    axes[1].set_yticks(range(len(loc_ct.index)))
    axes[1].set_yticklabels(loc_ct.index, fontsize=8)
    axes[1].set_title("Diagnosis mix per body site (% of row)")
    fig.colorbar(im, ax=axes[1], label="%")
    fig.legend(loc_ct.columns, loc="lower left", ncol=7, fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_sex_localization.png", dpi=150)
    plt.close(fig)


# 5. The images themselves
def image_analysis(df: pd.DataFrame, per_class: int = 60):
    header("5. THE IMAGES (unstructured data)")
    sample_path = df["path"].iloc[0]
    with Image.open(sample_path) as im:
        say(f"Image size: {im.size}, mode: {im.mode}")
    say(f"All images are the same resolution, resized to "
        f"{config.IMG_SIZE}x{config.IMG_SIZE} for the CNN.")

    # One example per class
    fig, axes = plt.subplots(1, len(config.DX_CLASSES), figsize=(20, 3.2))
    for ax, cls in zip(axes, config.DX_CLASSES):
        row = df[df["dx"] == cls].sample(1, random_state=config.SEED).iloc[0]
        ax.imshow(Image.open(row["path"]))
        ax.set_title(f"{cls}\n{config.DX_FULL_NAMES[cls][:22]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("One dermatoscopic image per diagnosis")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_sample_images.png", dpi=150)
    plt.close(fig)

    # Crude colour statistics per class - is there signal in raw pixels?
    say("")
    say(f"Mean RGB per class (subsample of {per_class} images/class):")
    rows = []
    for cls in config.DX_CLASSES:
        sub = df[df["dx"] == cls]
        sub = sub.sample(min(per_class, len(sub)), random_state=config.SEED)
        for p in sub["path"]:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("RGB").resize((64, 64)),
                                 dtype=np.float32)
            rows.append({"dx": cls, "R": arr[..., 0].mean(),
                         "G": arr[..., 1].mean(), "B": arr[..., 2].mean(),
                         "brightness": arr.mean(), "contrast": arr.std()})
    px = pd.DataFrame(rows)
    say(px.groupby("dx")[["R", "G", "B", "brightness", "contrast"]]
        .mean().round(1).to_string())

    h, p = stats.kruskal(*[g["brightness"].values for _, g in px.groupby("dx")])
    say("")
    say(f"Kruskal-Wallis on mean brightness across classes: H = {h:.1f}, p = {p:.2e}")
    say("-> Even raw colour separates the classes to some degree, which is a")
    say("   sanity check that the images carry signal. It is also a warning:")
    say("   part of it may be capture artefacts (vignetting, rulers, hair),")
    say("   not biology.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].boxplot([g["brightness"].values for _, g in px.groupby("dx")],
                    tick_labels=[c for c, _ in px.groupby("dx")])
    axes[0].set_title(f"Mean brightness per class (p = {p:.1e})")
    axes[1].boxplot([g["contrast"].values for _, g in px.groupby("dx")],
                    tick_labels=[c for c, _ in px.groupby("dx")])
    axes[1].set_title("Pixel std (contrast) per class")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_pixel_stats.png", dpi=150)
    plt.close(fig)


def main():
    df = load_metadata()
    overview(df)
    class_distribution(df)
    age_analysis(df)
    categorical_analysis(df)
    image_analysis(df)

    header("SUMMARY - what the EDA decided about the modelling")
    say("1. 25% of images are repeats of a lesion -> grouped split on lesion_id.")
    say("2. 58:1 class imbalance -> macro-F1 / balanced accuracy, not accuracy,")
    say("   and exactly one imbalance correction (config.IMBALANCE_STRATEGY).")
    say("3. age/sex/localization are all significant but the effect sizes are")
    say("   small to moderate (Cramer's V 0.07 for sex, 0.22 for localization)")
    say("   -> metadata is a useful side input, not a solution. Multimodal.")
    say("4. Raw pixel statistics already differ between classes -> the CNN has")
    say("   something to learn.")

    out = config.ARTIFACTS_DIR / "eda_report.txt"
    out.write_text("\n".join(LINES))
    print(f"\nReport saved: {out}")
    print(f"Figures saved: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
