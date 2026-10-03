from pathlib import Path
from huggingface_hub import snapshot_download
from PIL import Image
import shutil
import random


# ============================================================
# SETTINGS
# ============================================================

DATASET_NAME = "SobanHM/Road-Objects-Detection-Dataset"

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/dataset")

PEDESTRIAN_DIR = OUTPUT_DIR / "pedestrian"
TRAFFIC_SIGNAL_DIR = OUTPUT_DIR / "traffic_signal"

TARGET_PER_CLASS = 250

# Dataset class IDs
PERSON_ID = 3
TRAFFIC_SIGNAL_ID = 4

RANDOM_SEED = 42


# ============================================================
# DOWNLOAD
# ============================================================

def download_dataset():

    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print("Dataset already downloaded.")
        return

    print("Downloading dataset...")

    snapshot_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        local_dir=RAW_DIR
    )

    print("Download complete.")


# ============================================================
# READ YOLO LABEL
# ============================================================

def get_classes(label_file):

    classes = set()

    with open(label_file, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if not parts:
                continue

            class_id = int(parts[0])
            classes.add(class_id)

    return classes


# ============================================================
# FIND CANDIDATES
# ============================================================

def collect_candidates():

    pedestrian = []
    traffic_signal = []

    for split in ["train", "valid", "test"]:

        image_dir = RAW_DIR / split / "images"
        label_dir = RAW_DIR / split / "labels"

        if not image_dir.exists():
            continue

        for image_path in image_dir.iterdir():

            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            classes = get_classes(label_path)

            if PERSON_ID in classes:
                pedestrian.append(image_path)

            if TRAFFIC_SIGNAL_ID in classes:
                traffic_signal.append(image_path)

    return pedestrian, traffic_signal


# ============================================================
# COPY SELECTED IMAGES
# ============================================================

def prepare_output():

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    PEDESTRIAN_DIR.mkdir(parents=True)
    TRAFFIC_SIGNAL_DIR.mkdir(parents=True)


def copy_images(images, destination, prefix):

    random.shuffle(images)

    selected = images[:TARGET_PER_CLASS]

    for i, image_path in enumerate(selected):

        output_name = f"{prefix}_{i:04d}.jpg"
        output_path = destination / output_name

        try:

            image = Image.open(image_path).convert("RGB")
            image.save(output_path, quality=95)

        except Exception:
            continue

    return len(list(destination.glob("*.jpg")))


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(RANDOM_SEED)

    download_dataset()

    print("\nInspecting dataset...")

    pedestrian, traffic_signal = collect_candidates()

    print(f"Pedestrian candidates     : {len(pedestrian)}")
    print(f"Traffic Signal candidates: {len(traffic_signal)}")

    if len(pedestrian) < TARGET_PER_CLASS:
        raise RuntimeError("Not enough pedestrian images.")

    if len(traffic_signal) < TARGET_PER_CLASS:
        raise RuntimeError("Not enough traffic-signal images.")

    prepare_output()

    pedestrian_count = copy_images(
        pedestrian,
        PEDESTRIAN_DIR,
        "pedestrian"
    )

    traffic_signal_count = copy_images(
        traffic_signal,
        TRAFFIC_SIGNAL_DIR,
        "traffic_signal"
    )

    print("\n================================")
    print("CLEAN DATASET CREATED")
    print("================================")
    print(f"Pedestrian     : {pedestrian_count}")
    print(f"Traffic Signal : {traffic_signal_count}")
    print(f"Total          : {pedestrian_count + traffic_signal_count}")
    print("\nLabels:")
    print("0 = Pedestrian")
    print("1 = Traffic Signal")


if __name__ == "__main__":
    main()