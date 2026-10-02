from datasets import load_dataset
from pathlib import Path
import json


# ============================================================
# SETTINGS
# ============================================================

PEDESTRIAN_DATASET = "thirdeyelabs/indian-road-dataset"

MAX_PEDESTRIANS = 1000

MIN_CONFIDENCE = 0.70

# Keep frames separated to reduce repeated video frames.
# Example:
# frame 100 -> keep
# frame 101-109 -> skip
# frame 110 -> can keep
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

    # Traffic-sign directory is NOT recreated.
    # The existing 1000 verified images are kept.


# ============================================================
# PEDESTRIAN EXTRACTION
# ============================================================

def extract_pedestrians():

    print("\n==============================")
    print("EXTRACTING FULL PEDESTRIAN FRAMES")
    print("==============================")

    dataset = load_dataset(
        PEDESTRIAN_DATASET,
        split="train",
        streaming=True
    )

    count = 0

    metadata = []

    # Store last accepted frame for each clip.
    last_frame_by_clip = {}

    for sample in dataset:

        if count >= MAX_PEDESTRIANS:
            break

        # ----------------------------------------------------
        # Get image
        # ----------------------------------------------------

        image = sample.get("jpg")

        if image is None:
            image = sample.get("png")

        if image is None:
            continue

        # ----------------------------------------------------
        # Get annotation
        # ----------------------------------------------------

        annotation = sample["json"]

        frame_name = annotation.get(
            "name",
            sample["__key__"]
        )

        # ----------------------------------------------------
        # Get clip ID and filename
        # ----------------------------------------------------

        if "/" in frame_name:

            clip_id, filename = frame_name.rsplit(
                "/",
                1
            )

        else:

            clip_id = "unknown"
            filename = frame_name

        # ----------------------------------------------------
        # Get frame number
        # ----------------------------------------------------

        try:

            frame_number = int(
                Path(filename).stem
            )

        except ValueError:

            frame_number = None

        # ----------------------------------------------------
        # Get weather
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
        # Find a valid pedestrian
        # ----------------------------------------------------

        valid_pedestrian = False
        best_confidence = 0.0

        for label in annotation.get(
            "labels",
            []
        ):

            # We only want pedestrians.
            if label.get("category") != "person":
                continue

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = label.get(
                "confidence",
                1.0
            )

            if confidence < MIN_CONFIDENCE:
                continue

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            box = label.get("box2d")

            if not box:
                continue

            try:

                x1 = float(box["x1"])
                y1 = float(box["y1"])
                x2 = float(box["x2"])
                y2 = float(box["y2"])

            except (KeyError, TypeError, ValueError):

                continue

            # ------------------------------------------------
            # Check pedestrian size
            #
            # We are NOT cropping.
            #
            # We only use the box to make sure that the
            # pedestrian detection is meaningful.
            # ------------------------------------------------

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width < 20:
                continue

            if box_height < 30:
                continue

            valid_pedestrian = True

            best_confidence = max(
                best_confidence,
                confidence
            )

        # No valid pedestrian in this frame.
        if not valid_pedestrian:
            continue

        # ----------------------------------------------------
        # Avoid nearby repeated frames
        # ----------------------------------------------------

        if frame_number is not None:

            previous_frame = last_frame_by_clip.get(
                clip_id
            )

            if previous_frame is not None:

                frame_difference = (
                    frame_number - previous_frame
                )

                if frame_difference < MIN_FRAME_GAP:

                    continue

        # ----------------------------------------------------
        # SAVE FULL ORIGINAL IMAGE
        # ----------------------------------------------------

        output_name = (
            f"pedestrian_{count:04d}.jpg"
        )

        image.save(
            PEDESTRIAN_DIR / output_name,
            quality=90
        )

        # ----------------------------------------------------
        # Save metadata
        # ----------------------------------------------------

        metadata.append({

            "filename": output_name,

            "class": "pedestrian",

            "weather": weather,

            "clip_id": clip_id,

            "frame": frame_name,

            "frame_number": frame_number,

            "confidence": best_confidence

        })

        count += 1

        # Remember this frame for this clip.
        if frame_number is not None:

            last_frame_by_clip[
                clip_id
            ] = frame_number

        print(
            f"\rPedestrian frames: "
            f"{count}/{MAX_PEDESTRIANS}",
            end=""
        )

    print(
        "\nPedestrian extraction complete."
    )

    return metadata


# ============================================================
# READ EXISTING TRAFFIC-SIGN DATASET
# ============================================================

def read_existing_traffic_signs():

    print("\n==============================")
    print("READING EXISTING TRAFFIC SIGNS")
    print("==============================")

    metadata_path = (
        TRAFFIC_SIGN_DIR /
        "metadata.json"
    )

    metadata = []

    # --------------------------------------------------------
    # If metadata exists, use it.
    # --------------------------------------------------------

    if metadata_path.exists():

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

        print(
            f"Existing traffic-sign metadata: "
            f"{len(metadata)}"
        )

        return metadata

    # --------------------------------------------------------
    # If metadata does not exist, rebuild metadata from files.
    # --------------------------------------------------------

    image_files = sorted(
        TRAFFIC_SIGN_DIR.glob("*.jpg")
    )

    for image_path in image_files:

        metadata.append({

            "filename": image_path.name,

            "class": "traffic_sign",

            "sign_type": "unknown",

            "original_name": image_path.name,

            "weather": "unknown"

        })

    print(
        f"Traffic-sign images found: "
        f"{len(metadata)}"
    )

    return metadata


# ============================================================
# SAVE COMBINED METADATA
# ============================================================

def save_metadata(
    pedestrian_metadata,
    traffic_sign_metadata
):

    all_metadata = (
        pedestrian_metadata
        + traffic_sign_metadata
    )

    metadata_path = (
        OUTPUT_DIR /
        "metadata.json"
    )

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

    return metadata_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("BUILDING FINAL DATASET")
    print("========================================")

    # --------------------------------------------------------
    # Create required directories
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # Extract ONLY pedestrians
    # --------------------------------------------------------

    pedestrian_metadata = (
        extract_pedestrians()
    )

    # --------------------------------------------------------
    # KEEP existing traffic signs
    # --------------------------------------------------------

    traffic_sign_metadata = (
        read_existing_traffic_signs()
    )

    # --------------------------------------------------------
    # Combine metadata
    # --------------------------------------------------------

    metadata_path = save_metadata(
        pedestrian_metadata,
        traffic_sign_metadata
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n========================================")
    print("FINAL DATASET READY")
    print("========================================")

    print(
        f"Pedestrians   : "
        f"{len(pedestrian_metadata)}"
    )

    print(
        f"Traffic signs : "
        f"{len(traffic_sign_metadata)}"
    )

    print(
        f"Total samples : "
        f"{len(pedestrian_metadata) + len(traffic_sign_metadata)}"
    )

    print(
        f"Metadata      : "
        f"{metadata_path}"
    )

    print("\nTraffic-sign images were NOT downloaded again.")
    print("Existing verified traffic-sign images were kept.")


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()