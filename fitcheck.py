
# =============================================================================
# FitCheck — ML Model Fit Diagnostics
# Overfitting • Underfitting • Generalization Analysis
# =============================================================================
#
# Copyright (c) 2026 Naeim Mousavi
# All rights reserved.
#
# Description:
#     A lightweight diagnostic tool for identifying potential overfitting
#     and underfitting through model complexity analysis.
#
# Method:
#     Decision Tree with varying maximum tree depths.
#
# Evaluation:
#     Training Accuracy, Validation Accuracy, Generalization Gap.
#
# Outputs:
#     results/statistics.csv
#     results/summary.json
#     results/diagnostics.png
#
# Requirements:
#     numpy, pandas, matplotlib, scikit-learn
#
# Usage:
#     python fitcheck.py
#
# =============================================================================

print()
print("╭───────────────────────────────────────────────────────────────────╮")
print("│                         FitCheck                                  │")
print("│                  ML Model Fit Diagnostics                         │")
print("│                                                                   │")
print("│                         Decision Tree                             │")
print("│                                                                   │")
print("│          Overfitting • Underfitting • Generalization              │")
print("│                                                                   │")
print("│                 Copyright © 2026 Naeim Mousavi                    │")
print("│       For methodology and usage, please see the README.           │")
print("╰───────────────────────────────────────────────────────────────────╯")
print()

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# =============================================================================
# Configuration
# =============================================================================

RANDOM_STATE = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEPTHS = list(range(1, 21))
GAP_THRESHOLD = 0.10
LOW_ACCURACY_THRESHOLD = 0.75


# =============================================================================
# Dataset
# =============================================================================

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=2,
    n_redundant=2,
    n_classes=2,
    class_sep=0.8,
    flip_y=0.05,
    random_state=RANDOM_STATE,
)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y,
)


# =============================================================================
# Model Evaluation
# =============================================================================

train_scores = []
val_scores = []

for depth in DEPTHS:
    model = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    train_scores.append(
        accuracy_score(y_train, model.predict(X_train))
    )
    val_scores.append(
        accuracy_score(y_val, model.predict(X_val))
    )

train_scores = np.array(train_scores)
val_scores = np.array(val_scores)
generalization_gap = train_scores - val_scores


# =============================================================================
# Best Model Selection
# =============================================================================

# Choose the shallowest tree among those with the highest validation accuracy.
best_val = float(np.max(val_scores))
best_idx = int(np.flatnonzero(val_scores == best_val)[0])

best_depth = DEPTHS[best_idx]
best_train = float(train_scores[best_idx])
best_gap = float(generalization_gap[best_idx])


# =============================================================================
# Diagnostic
# =============================================================================

if (
    best_train < LOW_ACCURACY_THRESHOLD
    and best_val < LOW_ACCURACY_THRESHOLD
):
    diagnosis = "Possible Underfitting"
    explanation = (
        "Both training and validation accuracy are below the threshold."
    )

elif best_gap > GAP_THRESHOLD:
    diagnosis = "Possible Overfitting"
    explanation = (
        "Training accuracy substantially exceeds validation accuracy."
    )

else:
    diagnosis = "Good Fit"
    explanation = (
        "No strong warning from the selected heuristic thresholds."
    )


# =============================================================================
# Save Statistics
# =============================================================================

statistics = pd.DataFrame({
    "max_depth": DEPTHS,
    "train_accuracy": train_scores,
    "validation_accuracy": val_scores,
    "generalization_gap": generalization_gap,
})

statistics.to_csv(
    RESULTS_DIR / "statistics.csv",
    index=False,
    float_format="%.4f",
)

summary = {
    "project": "FitCheck",
    "model": "DecisionTreeClassifier",
    "diagnosis": diagnosis,
    "explanation": explanation,
    "best_max_depth": best_depth,
    "train_accuracy": round(best_train, 4),
    "validation_accuracy": round(best_val, 4),
    "generalization_gap": round(best_gap, 4),
    "thresholds": {
        "generalization_gap": GAP_THRESHOLD,
        "low_accuracy": LOW_ACCURACY_THRESHOLD,
    },
    "dataset": {
        "samples": len(X),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
    },
}

with open(
    RESULTS_DIR / "summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=4)


# =============================================================================
# Diagnostic Plots
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Training vs Validation Accuracy
axes[0].plot(
    DEPTHS,
    train_scores,
    marker="o",
    markersize=3,
    label="Train",
)
axes[0].plot(
    DEPTHS,
    val_scores,
    marker="o",
    markersize=3,
    label="Validation",
)
axes[0].axvline(
    best_depth,
    color="green",
    linestyle="--",
    alpha=0.7,
    label=f"Best depth: {best_depth}",
)
axes[0].set_title("Model Performance")
axes[0].set_xlabel("Maximum Tree Depth")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1.05)
axes[0].legend()
axes[0].grid(alpha=0.25)

# Plot 2: Generalization Gap
axes[1].plot(
    DEPTHS,
    generalization_gap,
    color="purple",
    marker="o",
    markersize=3,
)
axes[1].axhline(
    GAP_THRESHOLD,
    color="red",
    linestyle="--",
    label=f"Warning threshold: {GAP_THRESHOLD:.2f}",
)
axes[1].axhline(
    0,
    color="gray",
    linewidth=0.8,
)
axes[1].axvline(
    best_depth,
    color="green",
    linestyle="--",
    alpha=0.7,
)
axes[1].set_title("Generalization Gap")
axes[1].set_xlabel("Maximum Tree Depth")
axes[1].set_ylabel("Train Accuracy - Validation Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.25)

# Plot 3: Best Model
axes[2].bar(
    ["Train", "Validation"],
    [best_train, best_val],
    color=["royalblue", "darkorange"],
)
axes[2].set_title(f"Best Model: {diagnosis}")
axes[2].set_ylabel("Accuracy")
axes[2].set_ylim(0, 1.1)
axes[2].grid(axis="y", alpha=0.25)

for bar, value in zip(
    axes[2].patches,
    [best_train, best_val],
):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
    )

fig.suptitle(
    "FitCheck — Decision Tree Diagnostics",
    fontsize=14,
)
fig.tight_layout()

fig.savefig(
    RESULTS_DIR / "diagnostics.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# =============================================================================
# User-Friendly Console Report
# =============================================================================

print("\n" + "=" * 60)
print("              FitCheck | Diagnostic Report")
print("=" * 60)

print("\n[DIAGNOSIS]")
print(f"  Result: {diagnosis}")
print(f"  Details: {explanation}")

print("\n[MODEL PERFORMANCE]")
print(f"  Best max_depth:      {best_depth}")
print(f"  Training accuracy:   {best_train:.2%}")
print(f"  Validation accuracy: {best_val:.2%}")
print(f"  Generalization gap:  {best_gap:.2%}")

print("\n[CURRENT SETTINGS]")
print(f"  Dataset samples:       {len(X)}")
print(f"  Training samples:      {len(X_train)}")
print(f"  Validation samples:    {len(X_val)}")
print(f"  Tested tree depths:    {DEPTHS[0]} to {DEPTHS[-1]}")
print(f"  Minimum leaf samples:  5")
print(f"  Random state:          {RANDOM_STATE}")
print(f"  Overfitting threshold: {GAP_THRESHOLD:.0%}")
print(f"  Low accuracy limit:    {LOW_ACCURACY_THRESHOLD:.0%}")

print("\n[OUTPUT FILES]")
print(f"  Folder: {RESULTS_DIR}")
print("  statistics.csv  -> Metrics for every tested depth")
print("  summary.json    -> Diagnosis and model statistics")
print("  diagnostics.png -> Three diagnostic plots")

print("\n[IMPORTANT]")
print("  Thresholds provide heuristic warnings, not certainty.")
print("  Validation results may vary with the dataset and split.")
print("  Use cross-validation and a separate test set for robust evaluation.")

print("\n" + "=" * 60)
