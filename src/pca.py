from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models")

RANDOM_STATE = 42
TEST_SIZE = 0.20
VARIANCE_THRESHOLD = 0.95


# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)


# ============================================================
# STRATIFIED TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain/Test split:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# PCA
# ============================================================

print("\nFitting PCA on TRAINING data only...")

pca = PCA(
    n_components=VARIANCE_THRESHOLD,
    svd_solver="full",
    random_state=RANDOM_STATE
)

X_train_pca = pca.fit_transform(X_train)

# Transform test data using the PCA fitted on training data
X_test_pca = pca.transform(X_test)


# ============================================================
# EXPLAINED VARIANCE
# ============================================================

selected_components = pca.n_components_
cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

explained_variance = cumulative_variance[-1]


print("\n==============================")
print("PCA RESULTS")
print("==============================")

print("Original features:", X.shape[1])
print("Selected components:", selected_components)
print(
    "Cumulative explained variance:",
    f"{explained_variance:.4f}"
)

print("\nTransformed shapes:")
print("X_train_pca:", X_train_pca.shape)
print("X_test_pca :", X_test_pca.shape)


# ============================================================
# SAVE PCA MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    pca,
    MODEL_DIR / "pca.joblib"
)


# ============================================================
# SAVE TRAIN / TEST SPLIT
# ============================================================

np.savez(
    DATA_DIR / "train_test_split.npz",
    X_train_pca=X_train_pca,
    X_test_pca=X_test_pca,
    y_train=y_train,
    y_test=y_test
)


# ============================================================
# SAVE EXPLAINED VARIANCE DATA
# ============================================================

np.save(
    RESULTS_DIR / "pca_explained_variance.npy",
    cumulative_variance
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Save again after ensuring directory exists
np.save(
    RESULTS_DIR / "pca_explained_variance.npy",
    cumulative_variance
)


# ============================================================
# PLOT EXPLAINED VARIANCE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance
)

plt.axhline(
    VARIANCE_THRESHOLD,
    linestyle="--",
    label="95% variance"
)

plt.axvline(
    selected_components,
    linestyle="--",
    label=f"{selected_components} components"
)

plt.xlabel("Number of PCA Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "pca_explained_variance.png",
    dpi=300
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==============================")
print("PCA COMPLETE")
print("==============================")

print("PCA model saved to:")
print("models/pca.joblib")

print("\nSplit saved to:")
print("data/processed/train_test_split.npz")

print("\nPlot saved to:")
print("results/pca_explained_variance.png")