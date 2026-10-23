"""
DVC "prepare" stage: raw photos -> face crops -> train / val / test .pt files.

Run from the project root:
    python -m src.data.make_dataset          (or: dvc repro prepare)

How the split works (settings in params.yaml):
  * Friend classes (one burst each): files are sorted by burst number and cut
    into blocks, e.g. frames 1-35 train, 36-42 val, 43-50 test. Neighbouring
    burst frames are near-copies, so a random split would leak test images
    into training. Blocks mean val/test are angles the model never saw.
  * "unknown" (several people): split by person, e.g. 3 people train,
    1 val, 1 test. So test checks if a never-seen stranger is called unknown.

Outputs:
  data/{train,val,test}.pt   dict(images uint8 [N,3,S,S], labels, classes, paths)
  data/crops/<split>/<class>/*.jpg   the crops, so you can look at them
  reports/data_stats.json    counts per split/class + skipped files
  reports/figures/face_crops.png     sample grid for the report
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from tqdm import tqdm

from src.data.face import FaceDetector, crop_square

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BURST_RE = re.compile(r"TIMEBURST(\d+)", re.IGNORECASE)
SESSION_RE = re.compile(r"^IMG_(\d{8}_\d{6})")
SPLITS = ("train", "val", "test")


def session_of(p: Path) -> str:
    """Burst/session id from the file name, e.g. IMG_20261023_125215_... -> 20261023_125215."""
    m = SESSION_RE.match(p.stem)
    return m.group(1) if m else p.parent.name


def sort_key(p: Path):
    """Sort by session, then by burst number as a number (so 2 comes before 10)."""
    m = BURST_RE.search(p.stem)
    return (session_of(p), int(m.group(1)) if m else -1, p.name)


def block_split(files, val_frac, test_frac):
    n = len(files)
    n_test, n_val = round(n * test_frac), round(n * val_frac)
    n_train = n - n_val - n_test
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def group_split(files, counts, class_name):
    sessions = sorted({session_of(f) for f in files})
    need = sum(counts[s] for s in SPLITS)
    if len(sessions) != need:
        raise ValueError(
            f"'{class_name}' has {len(sessions)} people/sessions {sessions}, "
            f"but params.yaml group_split asks for {need}. Update group_split."
        )
    out, i = {}, 0
    for s in SPLITS:
        chosen = set(sessions[i:i + counts[s]])
        out[s] = [f for f in files if session_of(f) in chosen]
        i += counts[s]
    return out


def save_grid(samples, classes, path, thumb=96, per_class=8):
    """One row per class, evenly spaced train crops (covers all angles/people), for the report."""
    grid = Image.new("RGB", (thumb * per_class, thumb * len(classes)), "white")
    for r, name in enumerate(classes):
        imgs = samples.get(name, [])
        step = max(1, len(imgs) // per_class)
        for c, img in enumerate(imgs[::step][:per_class]):
            grid.paste(img.resize((thumb, thumb)), (c * thumb, r * thumb))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def main():
    params = yaml.safe_load(open("params.yaml"))
    dp, fp = params["data"], params["face"]

    raw_dir, out_dir = Path(dp["raw_dir"]), Path(dp["out_dir"])
    size = dp["img_size"]
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(fp["model_path"], fp["score_threshold"], fp["max_side"])
    classes = sorted(d.name for d in raw_dir.iterdir() if d.is_dir())

    # ---- decide which file goes to which split ----
    assignment = {s: [] for s in SPLITS}  # split -> [(path, label)]
    for label, name in enumerate(classes):
        files = sorted(
            (p for p in (raw_dir / name).rglob("*") if p.suffix.lower() in EXTS),
            key=sort_key,
        )
        if name in dp["group_split_classes"]:
            parts = group_split(files, dp["group_split"], name)
        else:
            parts = block_split(files, dp["val_frac"], dp["test_frac"])
        for s in SPLITS:
            assignment[s] += [(p, label) for p in parts[s]]

    # ---- detect, crop, save ----
    stats = {"classes": classes, "img_size": size, "splits": {}, "skipped": []}
    grid_samples = defaultdict(list)

    for split in SPLITS:
        images, labels, paths = [], [], []
        for p, label in tqdm(assignment[split], desc=f"{split:5s}"):
            try:
                img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            except Exception as e:
                stats["skipped"].append({"file": str(p), "reason": f"unreadable: {e}"})
                continue

            box = detector.largest_face(np.asarray(img))
            if box is None:
                stats["skipped"].append({"file": str(p), "reason": "no face found"})
                continue

            face = crop_square(img, box, fp["margin"]).resize((size, size), Image.BICUBIC)

            crop_path = out_dir / "crops" / split / classes[label] / f"{p.stem}.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            face.save(crop_path, quality=92)
            if split == "train":
                grid_samples[classes[label]].append(face)

            images.append(torch.from_numpy(np.array(face)).permute(2, 0, 1))
            labels.append(label)
            paths.append(str(p))

        if not images:
            raise RuntimeError(f"No faces found for the {split} split - check raw/ and the face model.")

        data = {
            "images": torch.stack(images),                  # uint8 [N, 3, S, S]
            "labels": torch.tensor(labels, dtype=torch.long),
            "classes": classes,
            "paths": paths,
        }
        torch.save(data, out_dir / f"{split}.pt")

        counts = torch.bincount(data["labels"], minlength=len(classes)).tolist()
        stats["splits"][split] = {"total": len(labels), **dict(zip(classes, counts))}
        print(f"  {split:5s} {tuple(data['images'].shape)}  per class: {dict(zip(classes, counts))}")

    Path("reports").mkdir(exist_ok=True)
    with open("reports/data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    save_grid(grid_samples, classes, Path("reports/figures/face_crops.png"))

    if stats["skipped"]:
        print(f"\n  Skipped {len(stats['skipped'])} file(s) - see reports/data_stats.json:")
        for s in stats["skipped"]:
            print(f"    {s['file']}  ({s['reason']})")
    print("\nDone. Look at data/crops/ and reports/figures/face_crops.png to check the crops.")


if __name__ == "__main__":
    main()