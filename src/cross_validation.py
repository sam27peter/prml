from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# LOAD ORIGINAL PREPROCESSED DATA
# ============================================================

X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")

print("Dataset:", X.shape)
print("Labels :", y.shape)


# ============================================================
# STRATIFIED K-FOLD
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


fold_results = []


# ============================================================
# CROSS-VALIDATION
# ============================================================

for fold, (train_idx, val_idx) in enumerate(
    skf.split(X, y),
    start=1
):

    print(f"\n========== FOLD {fold} ==========")

    X_train = X[train_idx]
    X_val = X[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    print("Training samples:", len(train_idx))
    print("Validation samples:", len(val_idx))

    # --------------------------------------------------------
    # PCA — FIT ONLY ON TRAINING FOLD
    # --------------------------------------------------------

    pca = PCA(
        n_components=0.95,
        svd_solver="full"
    )

    X_train_pca = pca.fit_transform(X_train)

    X_val_pca = pca.transform(X_val)

    print(
        "PCA components:",
        pca.n_components_
    )

    # --------------------------------------------------------
    # MLP — SGD
    # --------------------------------------------------------

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="sgd",
        learning_rate_init=0.01,
        momentum=0.95,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train_pca,
        y_train
    )

    # --------------------------------------------------------
    # VALIDATION PREDICTION
    # --------------------------------------------------------

    y_pred = model.predict(X_val_pca)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )

    fold_results.append(
        [accuracy, precision, recall, f1]
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")


# ============================================================
# RESULTS
# ============================================================

results = np.asarray(
    fold_results
)

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

means = results.mean(axis=0)
stds = results.std(axis=0)


print("\n==============================")
print("STRATIFIED K-FOLD RESULTS")
print("==============================")

for name, mean, std in zip(
    metric_names,
    means,
    stds
):

    print(
        f"{name:<10}: "
        f"{mean:.4f} ± {std:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    RESULTS_DIR / "cross_validation_results.txt",
    "w"
) as f:

    f.write("STRATIFIED 5-FOLD CROSS-VALIDATION\n")
    f.write("=================================\n\n")

    for i, row in enumerate(
        results,
        start=1
    ):

        f.write(
            f"Fold {i}: "
            f"Accuracy={row[0]:.4f}, "
            f"Precision={row[1]:.4f}, "
            f"Recall={row[2]:.4f}, "
            f"F1={row[3]:.4f}\n"
        )

    f.write("\nMean ± Standard Deviation\n\n")

    for name, mean, std in zip(
        metric_names,
        means,
        stds
    ):

        f.write(
            f"{name}: "
            f"{mean:.4f} ± {std:.4f}\n"
        )


print("\nResults saved to:")
print("results/cross_validation_results.txt")   