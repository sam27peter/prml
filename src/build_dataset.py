from datasets import load_dataset
from pathlib import Path
from PIL import Image
import json


# ============================================================
# SETTINGS
# ============================================================

PEDESTRIAN_DATASET = "thirdeyelabs/indian-road-dataset"
TRAFFIC_SIGN_DATASET = "chandrabhuma/Indian_Traffic_VQA_Dataset"

MAX_PEDESTRIANS = 1000
MAX_TRAFFIC_SIGNS = 1000

MIN_CONFIDENCE = 0.70

# Keep frames separated to reduce repeated video-frame samples.
# Example: if we keep frame 100, skip nearby frames until 110.
MIN_FRAME_GAP = 10

OUTPUT_DIR = Path("data/dataset")

PEDESTRIAN_DIR = OUTPUT_DIR / "pedestrian"
TRAFFIC_SIGN_DIR = OUTPUT_DIR / "traffic_sign"


# ============================================================
# FOLDER SETUP
# ============================================================

def create_directories():

    PEDESTRIAN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TRAFFIC_SIGN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# PEDESTRIAN EXTRACTION
# ============================================================

def extract_pedestrians():

    print("\n==============================")
    print("EXTRACTING PEDESTRIANS")
    print("==============================")

    dataset = load_dataset(
        PEDESTRIAN_DATASET,
        split="train",
        streaming=True
    )

    count = 0

    metadata = []

    # Store the last accepted frame for every video clip.
    last_frame_by_clip = {}

    for sample in dataset:

        if count >= MAX_PEDESTRIANS:
            break

        image = sample.get("jpg")

        if image is None:
            image = sample.get("png")

        if image is None:
            continue

        annotation = sample["json"]

        frame_name = annotation.get(
            "name",
            sample["__key__"]
        )

        # ----------------------------------------------------
        # Clip ID
        # ----------------------------------------------------

        if "/" in frame_name:
            clip_id, filename = frame_name.rsplit("/", 1)
        else:
            clip_id = "unknown"
            filename = frame_name

        # ----------------------------------------------------
        # Frame number
        # ----------------------------------------------------

        try:
            frame_number = int(
                Path(filename).stem
            )
        except ValueError:
            frame_number = None

        # ----------------------------------------------------
        # Weather
        # ----------------------------------------------------

        attributes = annotation.get(
            "attributes",
            {}
        )

        weather = attributes.get(
            "weather",
            "unknown"
        )

        # ----------------------------------------------------
        # Check labels
        # ----------------------------------------------------

        for label in annotation.get("labels", []):

            if count >= MAX_PEDESTRIANS:
                break

            if label.get("category") != "person":
                continue

            confidence = label.get(
                "confidence",
                1.0
            )

            if confidence < MIN_CONFIDENCE:
                continue

            # ------------------------------------------------
            # Avoid repeated nearby frames
            # ------------------------------------------------

            if frame_number is not None:

                previous_frame = last_frame_by_clip.get(
                    clip_id
                )

                if (
                    previous_frame is not None
                    and frame_number - previous_frame
                    < MIN_FRAME_GAP
                ):
                    continue

            box = label.get("box2d")

            if not box:
                continue

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1 = int(float(box["x1"]))
            y1 = int(float(box["y1"]))
            x2 = int(float(box["x2"]))
            y2 = int(float(box["y2"]))

            image_width, image_height = image.size

            # Coordinates must stay inside image.
            x1 = max(0, min(x1, image_width))
            x2 = max(0, min(x2, image_width))

            y1 = max(0, min(y1, image_height))
            y2 = max(0, min(y2, image_height))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop(
                (x1, y1, x2, y2)
            )

            # Reject extremely small people.
            if crop.width < 20 or crop.height < 30:
                continue

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            output_name = (
                f"pedestrian_{count:04d}.jpg"
            )

            crop.save(
                PEDESTRIAN_DIR / output_name,
                quality=95
            )

            metadata.append({
                "filename": output_name,
                "class": "pedestrian",
                "weather": weather,
                "clip_id": clip_id,
                "frame": frame_name,
                "frame_number": frame_number,
                "confidence": confidence
            })

            count += 1

            if frame_number is not None:
                last_frame_by_clip[clip_id] = frame_number

            print(
                f"\rPedestrians: {count}/{MAX_PEDESTRIANS}",
                end=""
            )

    print("\nPedestrian extraction complete.")

    return metadata


# ============================================================
# TRAFFIC SIGN EXTRACTION
# ============================================================

def extract_traffic_signs():

    print("\n==============================")
    print("EXTRACTING TRAFFIC SIGNS")
    print("==============================")

    dataset = load_dataset(
        TRAFFIC_SIGN_DATASET,
        split="train",
        streaming=True
    )

    count = 0

    metadata = []

    # Prevent the same image appearing more than once.
    seen_images = set()

    for sample in dataset:

        if count >= MAX_TRAFFIC_SIGNS:
            break

        image_name = sample["image_name"]

        if image_name in seen_images:
            continue

        seen_images.add(image_name)

        image = sample["image"]

        if image is None:
            continue

        sign_type = sample[" traffic_sign"]

        output_name = (
            f"traffic_sign_{count:04d}.jpg"
        )

        image.save(
            TRAFFIC_SIGN_DIR / output_name,
            quality=95
        )

        metadata.append({
            "filename": output_name,
            "class": "traffic_sign",
            "sign_type": sign_type,
            "original_name": image_name,
            "weather": "unknown"
        })

        count += 1

        print(
            f"\rTraffic signs: {count}/{MAX_TRAFFIC_SIGNS}",
            end=""
        )

    print("\nTraffic-sign extraction complete.")

    return metadata


# ============================================================
# MAIN
# ============================================================

def main():

    create_directories()

    pedestrian_metadata = extract_pedestrians()

    traffic_sign_metadata = extract_traffic_signs()

    all_metadata = (
        pedestrian_metadata
        + traffic_sign_metadata
    )

    metadata_path = OUTPUT_DIR / "metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_metadata,
            file,
            indent=2
        )

    print("\n==============================")
    print("DATASET BUILD COMPLETE")
    print("==============================")

    print(
        f"Pedestrians   : {len(pedestrian_metadata)}"
    )

    print(
        f"Traffic signs : {len(traffic_sign_metadata)}"
    )

    print(
        f"Total samples : {len(all_metadata)}"
    )

    print(
        f"Metadata      : {metadata_path}"
    )


if __name__ == "__main__":
    main()