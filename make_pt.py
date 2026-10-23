"""
Build train / val / test .pt files from a folder of class sub-folders.

Expected layout (run this script from the ML02 folder):

    ML02/
      make_pt.py
      raw/
        dhamith/  pawara/  uchitha/  unknown/

Output:

    ML02/data/train.pt
    ML02/data/val.pt
    ML02/data/test.pt

Each .pt file is a dict:
    images  -> uint8 tensor  [N, 3, IMG_SIZE, IMG_SIZE]
    labels  -> int64 tensor  [N]
    classes -> list of class names (index = label)
    paths   -> list of the original file paths (handy for debugging)
"""
from pathlib import Path

import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm

# ---------------- settings ----------------
RAW_DIR = Path("raw")
OUT_DIR = Path("data")
IMG_SIZE = 224          # 224 for ResNet/MobileNet etc. Use 128 if RAM is tight.
TEST_SIZE = 0.15        # 15% test
VAL_SIZE = 0.15         # 15% validation  -> 70% train
SEED = 42
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ------------------------------------------


def collect_files():
    classes = sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir())
    paths, labels = [], []
    print("Images per class:")
    for idx, name in enumerate(classes):
        files = sorted(
            p for p in (RAW_DIR / name).rglob("*") if p.suffix.lower() in EXTS
        )
        print(f"  [{idx}] {name:10s} {len(files)}")
        paths += files
        labels += [idx] * len(files)
    print(f"  total      {len(paths)}\n")
    return classes, paths, labels


to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.PILToTensor(),            # uint8, [3, H, W], values 0-255
])


def build_split(paths, labels, classes, split_name):
    images, ys, kept, skipped = [], [], [], []
    for p, y in tqdm(list(zip(paths, labels)), desc=f"{split_name:5s}"):
        try:
            img = Image.open(p)
            img = ImageOps.exif_transpose(img)   # fix phone-photo rotation
            img = img.convert("RGB")             # drop alpha / grayscale -> 3 channels
        except Exception:
            skipped.append(str(p))
            continue
        images.append(to_tensor(img))
        ys.append(y)
        kept.append(str(p))

    data = {
        "images": torch.stack(images),
        "labels": torch.tensor(ys, dtype=torch.long),
        "classes": classes,
        "paths": kept,
    }
    out = OUT_DIR / f"{split_name}.pt"
    torch.save(data, out)

    counts = torch.bincount(data["labels"], minlength=len(classes)).tolist()
    print(f"  saved {out}  shape={tuple(data['images'].shape)}  per-class={counts}")
    if skipped:
        print(f"  skipped {len(skipped)} unreadable file(s):")
        for s in skipped:
            print("   ", s)
    return data


def main():
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    classes, paths, labels = collect_files()

    # 1) split off the test set
    trval_p, test_p, trval_y, test_y = train_test_split(
        paths, labels, test_size=TEST_SIZE, stratify=labels, random_state=SEED
    )
    # 2) split the rest into train / val
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    train_p, val_p, train_y, val_y = train_test_split(
        trval_p, trval_y, test_size=val_frac, stratify=trval_y, random_state=SEED
    )

    build_split(train_p, train_y, classes, "train")
    build_split(val_p, val_y, classes, "val")
    build_split(test_p, test_y, classes, "test")
    print("\nDone.")


if __name__ == "__main__":
    main()
