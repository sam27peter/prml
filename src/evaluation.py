from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

data = np.load(
    DATA_DIR / "train_test_split.npz"
)

X_test = data["X_test_pca"]
y_test = data["y_test"]


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_DIR / "mlp_sgd.joblib"
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

# Probability of class 1 = Pedestrian
y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

average_precision = average_precision_score(
    y_test,
    y_prob
)


print("\n==============================")
print("TEST SET EVALUATION")
print("==============================")

print("Test samples:", len(y_test))

print("\nMetrics:")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1 Score:          {f1:.4f}")
print(f"ROC-AUC:           {roc_auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Traffic Sign (Proxy)",
            "Pedestrian"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Traffic Sign (Proxy)",
        "Pedestrian"
    ]
)

disp.plot()

plt.title("Confusion Matrix")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "roc_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision_values, recall_values, _ = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    recall_values,
    precision_values,
    label=f"Average Precision = {average_precision:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "precision_recall_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE METRICS
# ============================================================

with open(
    RESULTS_DIR / "evaluation_metrics.txt",
    "w"
) as f:

    f.write("TEST SET EVALUATION\n")
    f.write("===================\n\n")

    f.write(f"Test samples: {len(y_test)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")
    f.write(
        f"Average Precision: "
        f"{average_precision:.4f}\n"
    )

print("\nPlots saved:")
print("results/confusion_matrix.png")
print("results/roc_curve.png")
print("results/precision_recall_curve.png")
print("results/evaluation_metrics.txt")