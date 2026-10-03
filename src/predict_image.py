from pathlib import Path

import joblib
import numpy as np

from preprocessing import preprocess_image


# ============================================================
# PATHS
# ============================================================

TEST_DIR = Path("test_images")

PCA_PATH = Path("models/pca.joblib")
MODEL_PATH = Path("models/mlp_sgd.joblib")


# ============================================================
# LOAD MODELS
# ============================================================

pca = joblib.load(PCA_PATH)
model = joblib.load(MODEL_PATH)


# ============================================================
# PREDICT ONE IMAGE
# ============================================================

def predict_image(image_path):

    # Same preprocessing used during training
    features = preprocess_image(image_path)

    # Convert to batch format
    features = features.reshape(1, -1)

    # PCA transformation
    features_pca = pca.transform(features)

    # Prediction
    prediction = model.predict(features_pca)[0]

    # Probability
    probabilities = model.predict_proba(features_pca)[0]

    if prediction == 1:
        label = "Pedestrian"
    else:
        label = "Traffic Sign (Proxy)"

    confidence = probabilities[prediction]

    print(f"\nImage: {image_path.name}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n==============================")
    print("IMAGE PREDICTION")
    print("==============================")

    images = sorted(
        p for p in TEST_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png"
        }
    )

    if not images:
        raise FileNotFoundError(
            "No test images found in test_images/"
        )

    for image_path in images:
        predict_image(image_path)

    print("\n==============================")
    print("PREDICTION COMPLETE")
    print("==============================")