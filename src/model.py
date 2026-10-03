from pathlib import Path

import joblib
import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

RANDOM_STATE = 42


# ============================================================
# LOAD PCA DATA
# ============================================================

data = np.load(
    DATA_DIR / "train_test_split.npz"
)

X_train = data["X_train_pca"]
y_train = data["y_train"]

print("Training data:", X_train.shape)
print("Training labels:", y_train.shape)


# ============================================================
# MLP WITH SGD
# ============================================================

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="sgd",
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=RANDOM_STATE
)


# ============================================================
# LEARNING RATE + MOMENTUM TUNING
# ============================================================

parameter_grid = {
    "learning_rate_init": [0.001, 0.01],
    "momentum": [0.8, 0.9, 0.95]
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=parameter_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining MLP with SGD...")
print("Tuning learning rate and momentum...")

grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_

print("\n==============================")
print("MLP TRAINING COMPLETE")
print("==============================")

print("Best parameters:")
print(grid_search.best_params_)

print(
    "Best CV F1:",
    f"{grid_search.best_score_:.4f}"
)

print(
    "Training iterations:",
    best_model.n_iter_
)

print(
    "Final training loss:",
    f"{best_model.loss_:.6f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    best_model,
    MODEL_DIR / "mlp_sgd.joblib"
)

print("\nModel saved to:")
print("models/mlp_sgd.joblib")