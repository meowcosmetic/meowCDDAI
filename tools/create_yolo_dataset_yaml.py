"""
Tạo file dataset.yaml cho Ultralytics YOLO từ folder dataset + classes.txt.

Giả định structure:
  dataset_root/
    train/images
    train/labels
    val/images   (optional)
    val/labels   (optional)
    classes.txt

Chạy:
  python tools/create_yolo_dataset_yaml.py --dataset dataset_root --out dataset_root/dataset.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def load_classes(classes_path: Path) -> List[str]:
    lines = [ln.strip() for ln in classes_path.read_text(encoding="utf-8").splitlines()]
    classes = [ln for ln in lines if ln]
    if not classes:
        raise ValueError("classes.txt rỗng")
    return classes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset root")
    ap.add_argument("--out", required=True, help="Path dataset.yaml output")
    args = ap.parse_args()

    root = Path(args.dataset)
    out = Path(args.out)
    classes_path = root / "classes.txt"
    if not classes_path.exists():
        print(f"❌ Không tìm thấy classes.txt: {classes_path}")
        return 1

    classes = load_classes(classes_path)

    # dùng absolute path để tránh nhầm cwd khi train
    root_abs = root.resolve()
    train_images = root_abs / "train" / "images"
    val_images = root_abs / "val" / "images"

    if not train_images.exists():
        print(f"❌ Không tìm thấy train/images: {train_images}")
        return 1

    yaml_lines = []
    yaml_lines.append(f"path: {root_abs.as_posix()}")
    yaml_lines.append("train: train/images")
    yaml_lines.append("val: val/images" if val_images.exists() else "val: train/images")
    yaml_lines.append("")
    yaml_lines.append("names:")
    for i, name in enumerate(classes):
        # yaml safe: quote
        safe = name.replace('"', '\\"')
        yaml_lines.append(f'  {i}: "{safe}"')

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"✅ Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


