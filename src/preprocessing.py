from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageOps


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_DIR = Path("data/dataset")
METADATA_PATH = DATASET_DIR / "metadata.json"

CANVAS_SIZE = (256, 256)


# --------------------------------------------------
# Metadata
# --------------------------------------------------

def load_metadata():
    """Load dataset metadata."""

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def load_image(image_path):
    """
    Load an image and correct its camera orientation.

    No cropping.
    No stretching.
    """

    with Image.open(image_path) as image:

        # Correct EXIF orientation
        image = ImageOps.exif_transpose(image)

        # Convert to RGB first
        image = image.convert("RGB")

        return image.copy()


def preserve_aspect_ratio(image):
    """
    Resize the COMPLETE image so that it fits inside
    the 256x256 canvas without changing its aspect ratio.

    The image is never cropped or stretched.
    """

    canvas_width, canvas_height = CANVAS_SIZE

    original_width, original_height = image.size

    scale = min(
        canvas_width / original_width,
        canvas_height / original_height
    )

    new_width = max(1, int(original_width * scale))
    new_height = max(1, int(original_height * scale))

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )


def create_canvas(image):
    """
    Place the complete image in the center of a
    fixed-size canvas.

    Remaining area is padding.
    """

    canvas_width, canvas_height = CANVAS_SIZE

    canvas = Image.new(
        "RGB",
        CANVAS_SIZE,
        color=(0, 0, 0)
    )

    image_width, image_height = image.size

    x = (canvas_width - image_width) // 2
    y = (canvas_height - image_height) // 2

    canvas.paste(image, (x, y))

    return canvas


def convert_to_grayscale(image):
    """Convert RGB image to grayscale."""

    return image.convert("L")


def normalize_image(image):
    """
    Convert pixels from [0, 255] to [0, 1].
    """

    array = np.asarray(
        image,
        dtype=np.float32
    )

    return array / 255.0


def flatten_image(image):
    """
    Convert 2D image into a 1D feature vector.
    """

    return image.reshape(-1)


# --------------------------------------------------
# Complete preprocessing pipeline
# --------------------------------------------------

def preprocess_image(image_path):
    """
    Complete preprocessing pipeline.

    Original image
        ↓
    EXIF correction
        ↓
    Aspect-ratio preserving resize
        ↓
    Padding
        ↓
    Grayscale
        ↓
    Normalization
        ↓
    Flattening
    """

    image = load_image(image_path)

    image = preserve_aspect_ratio(image)

    image = create_canvas(image)

    image = convert_to_grayscale(image)

    image = normalize_image(image)

    image = flatten_image(image)

    return image


# --------------------------------------------------
# Dataset processing
# --------------------------------------------------

def build_feature_matrix(metadata):
    """
    Convert all images into a feature matrix.

    Returns:
        X -> image features
        y -> class labels
    """

    X = []
    y = []

    label_map = {
        "pedestrian": 0,
        "traffic_sign": 1
    }

    total = len(metadata)

    for index, item in enumerate(metadata, start=1):

        image_path = (
            DATASET_DIR
            / item["class"]
            / item["filename"]
        )

        features = preprocess_image(image_path)

        X.append(features)
        y.append(label_map[item["class"]])

        print(
            f"\rProcessing images: "
            f"{index}/{total}",
            end=""
        )

    print()

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64)
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("\n================================")
    print("PREPROCESSING DATASET")
    print("================================")

    metadata = load_metadata()

    print(f"Total images: {len(metadata)}")
    print(f"Canvas size : {CANVAS_SIZE}")

    X, y = build_feature_matrix(metadata)

    print("\n================================")
    print("PREPROCESSING COMPLETE")
    print("================================")

    print("Feature matrix shape:", X.shape)
    print("Label vector shape  :", y.shape)

    print(
        "Features per image  :",
        X.shape[1]
    )

    print(
        "Memory used:",
        f"{X.nbytes / (1024 ** 2):.2f} MB"
    )