from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = Path("data/dataset")
OUTPUT_DIR = Path("data/processed")

CANVAS_SIZE = (256, 256)

# 0 = Not Pedestrian / Traffic Sign Proxy
# 1 = Pedestrian
CLASS_MAP = {
    "traffic_sign": 0,
    "pedestrian": 1,
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(image_path):
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image.copy()


def preserve_aspect_ratio(image):
    width, height = image.size
    canvas_width, canvas_height = CANVAS_SIZE

    scale = min(
        canvas_width / width,
        canvas_height / height
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )


def create_canvas(image):
    canvas = Image.new(
        "RGB",
        CANVAS_SIZE,
        color=(0, 0, 0)
    )

    width, height = image.size

    x = (CANVAS_SIZE[0] - width) // 2
    y = (CANVAS_SIZE[1] - height) // 2

    canvas.paste(image, (x, y))

    return canvas


def preprocess_image(image_path):
    image = load_image(image_path)

    # Preserve aspect ratio and resize
    image = preserve_aspect_ratio(image)

    # Pad to 256 x 256
    image = create_canvas(image)

    # Grayscale
    image = image.convert("L")

    # Normalize 0-255 -> 0-1
    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # 256 x 256 -> 65,536 features
    return image.reshape(-1)


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset():
    X = []
    y = []

    for class_name, label in CLASS_MAP.items():

        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing folder: {class_dir}"
            )

        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in VALID_EXTENSIONS
        )

        print(f"{class_name}: {len(images)} images")

        for image_path in images:
            features = preprocess_image(image_path)

            X.append(features)
            y.append(label)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    return X, y


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n==============================")
    print("PREPROCESSING DATASET")
    print("==============================")

    X, y = build_dataset()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)

    print("\n==============================")
    print("PREPROCESSING COMPLETE")
    print("==============================")

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Features per image:", X.shape[1])

    print("\nClass counts:")
    print("0 = Traffic Sign:", np.sum(y == 0))
    print("1 = Pedestrian:", np.sum(y == 1))