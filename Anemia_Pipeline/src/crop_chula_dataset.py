"""
Crop individually labeled RBCs out of the Chula-RBC-12 blood smear images
into per-class image folders, ready for a standard image-classification
training pipeline.

The dataset provides only a CENTER POINT (x, y) per cell, not a bounding
box, so this crops a fixed-size square patch centered on each point.

IMPORTANT — check the crop size visually before running the full dataset:
    python -m src.crop_chula_dataset --dataset-dir Chula-RBC-12-Dataset \
        --output-dir ./chula_cropped --crop-size 64 --preview-only

This saves a handful of sample crops to <output-dir>/_preview/ so you can
open them and confirm whole RBCs are visible (not cut off, not mostly
background). Adjust --crop-size up/down and re-run --preview-only until it
looks right, THEN run the full crop (drop --preview-only).

Usage (full run):
    python -m src.crop_chula_dataset --dataset-dir Chula-RBC-12-Dataset \
        --output-dir ./chula_cropped --crop-size 64
"""

import argparse
import os
import random

from PIL import Image

CLASS_NAMES = {
    0: "Normal_cell",
    1: "Macrocyte",
    2: "Microcyte",
    3: "Spherocyte",
    4: "Target_cell",
    5: "Stomatocyte",
    6: "Ovalocyte",
    7: "Teardrop",
    8: "Burr_cell",
    9: "Schistocyte",
    10: "uncategorised",
    11: "Hypochromia",
    12: "Elliptocyte",
}


def parse_label_file(label_path):
    """Yields (x, y, class_id) tuples from a Chula-RBC-12 label .txt file."""
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                print(f"Skipping malformed line in {label_path}: {line!r}")
                continue
            x, y, cls = parts
            yield int(float(x)), int(float(y)), int(cls)


def crop_cell(img, x, y, crop_size):
    half = crop_size // 2
    left, top = x - half, y - half
    right, bottom = x + half, y + half
    # Pad with black if the crop would go outside the image bounds
    # (Image.crop handles this automatically for out-of-bounds coordinates,
    # filling with black — acceptable for cells near the smear edge)
    return img.crop((left, top, right, bottom))


def main(dataset_dir, output_dir, crop_size, preview_only, exclude_classes):
    dataset_path = os.path.join(dataset_dir, "Dataset")
    label_path = os.path.join(dataset_dir, "Label")

    exclude_classes = set(exclude_classes or [])
    counts = {cls: 0 for cls in CLASS_NAMES}

    if preview_only:
        preview_dir = os.path.join(output_dir, "_preview")
        os.makedirs(preview_dir, exist_ok=True)
        label_files = sorted(os.listdir(label_path))
        random.seed(42)
        sample_files = random.sample(label_files, min(3, len(label_files)))
        n_saved = 0
        for lf in sample_files:
            img_id = os.path.splitext(lf)[0]
            img_file = os.path.join(dataset_path, f"{img_id}.jpg")
            if not os.path.exists(img_file):
                continue
            img = Image.open(img_file).convert("RGB")
            for i, (x, y, cls) in enumerate(parse_label_file(os.path.join(label_path, lf))):
                if i >= 5:  # a handful of crops per image is enough to check
                    break
                crop = crop_cell(img, x, y, crop_size)
                crop.save(os.path.join(preview_dir, f"{img_id}_{i}_{CLASS_NAMES[cls]}.png"))
                n_saved += 1
        print(f"Saved {n_saved} preview crops to {preview_dir}")
        print("Open a few of these and confirm whole RBCs are visible and centered.")
        print("If cells look too small/cut off, increase --crop-size; if mostly background, decrease it.")
        return

    for cls_id, cls_name in CLASS_NAMES.items():
        if cls_id in exclude_classes:
            continue
        os.makedirs(os.path.join(output_dir, cls_name), exist_ok=True)

    label_files = sorted(os.listdir(label_path))
    for lf in label_files:
        img_id = os.path.splitext(lf)[0]
        img_file = os.path.join(dataset_path, f"{img_id}.jpg")
        if not os.path.exists(img_file):
            print(f"Warning: no matching image for {lf}, skipping")
            continue

        img = Image.open(img_file).convert("RGB")
        for i, (x, y, cls) in enumerate(parse_label_file(os.path.join(label_path, lf))):
            if cls in exclude_classes:
                continue
            crop = crop_cell(img, x, y, crop_size)
            cls_name = CLASS_NAMES.get(cls)
            if cls_name is None:
                print(f"Unknown class id {cls} in {lf}, skipping")
                continue
            out_path = os.path.join(output_dir, cls_name, f"{img_id}_{i}.png")
            crop.save(out_path)
            counts[cls] += 1

    print("\n=== Crop counts per class ===")
    for cls_id, cls_name in CLASS_NAMES.items():
        if cls_id in exclude_classes:
            continue
        print(f"{cls_name}: {counts[cls_id]}")
    print(f"\nSaved cropped cells to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Path to cloned Chula-RBC-12-Dataset repo")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--preview-only", action="store_true",
                         help="Save a small preview set to check crop size before the full run")
    parser.add_argument("--exclude-classes", type=int, nargs="*", default=[10],
                         help="Class IDs to exclude (default: excludes 10=uncategorised, "
                              "which is not a real diagnostic category)")
    args = parser.parse_args()
    main(args.dataset_dir, args.output_dir, args.crop_size, args.preview_only, args.exclude_classes)
