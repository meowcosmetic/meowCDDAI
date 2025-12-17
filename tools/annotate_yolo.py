"""
Tool annotate bounding boxes cho ảnh và lưu label theo format YOLO.

Dataset structure đề xuất:
  dataset/
    images/  (input)
    labels/  (output .txt YOLO)
    classes.txt (mỗi dòng 1 class name)

Chạy:
  python tools/annotate_yolo.py --images dataset/images --labels dataset/labels --classes dataset/classes.txt

Phím tắt:
  - Chuột trái kéo-thả: vẽ bbox
  - 1..9: chọn class id 0..8 (id = phím-1)
  - 0: chọn class id 9 (nếu có)
  - [: class -1, ]: class +1
  - s: lưu label ảnh hiện tại
  - n / Right Arrow: ảnh tiếp
  - b / Left Arrow: ảnh trước
  - d: xoá bbox cuối cùng
  - c: hiện class đang chọn
  - h: toggle help overlay
  - q / ESC: thoát
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Box:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    def normalized_yolo(self, w: int, h: int) -> Tuple[int, float, float, float, float]:
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0
        return (
            self.class_id,
            cx / w,
            cy / h,
            bw / w,
            bh / h,
        )


def load_classes(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy classes file: {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    classes = [ln for ln in lines if ln]
    if not classes:
        raise ValueError("classes.txt rỗng")
    return classes


def list_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    imgs.sort(key=lambda p: p.name)
    return imgs


def read_yolo_labels(label_path: Path, w: int, h: int) -> List[Box]:
    if not label_path.exists():
        return []
    boxes: List[Box] = []
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        boxes.append(Box(cid, x1, y1, x2, y2))
    return boxes


def write_yolo_labels(label_path: Path, boxes: List[Box], w: int, h: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for b in boxes:
        cid, cx, cy, bw, bh = b.normalized_yolo(w, h)
        # clamp để tránh <0 hoặc >1 do kéo ra ngoài ảnh
        cx = min(1.0, max(0.0, cx))
        cy = min(1.0, max(0.0, cy))
        bw = min(1.0, max(0.0, bw))
        bh = min(1.0, max(0.0, bh))
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def draw_overlay(
    img: np.ndarray,
    boxes: List[Box],
    classes: List[str],
    current_class: int,
    idx: int,
    total: int,
    help_on: bool,
) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    # header
    header = f"[{idx+1}/{total}] class={current_class}:{classes[current_class] if 0 <= current_class < len(classes) else 'N/A'} | s=save n/b=next/prev d=del q=quit h=help"
    cv2.rectangle(out, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.putText(out, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # boxes
    for b in boxes:
        x1, x2 = sorted((b.x1, b.x2))
        y1, y2 = sorted((b.y1, b.y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        name = classes[b.class_id] if 0 <= b.class_id < len(classes) else str(b.class_id)
        cv2.putText(out, name, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    if help_on:
        help_lines = [
            "Mouse: Left drag = draw bbox",
            "Keys: 1..9 -> class 0..8, 0 -> class 9",
            "      [ / ] -> prev/next class",
            "      s save | n next | b prev | d delete last | q/ESC quit",
        ]
        y = 44
        for ln in help_lines:
            cv2.putText(out, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
            y += 20
    return out


class Annotator:
    def __init__(self, images: List[Path], labels_dir: Path, classes: List[str]):
        self.images = images
        self.labels_dir = labels_dir
        self.classes = classes
        self.current_class = 0
        self.help_on = True

        self.idx = 0
        self.boxes: List[Box] = []
        self.dragging = False
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_end: Optional[Tuple[int, int]] = None

        self.img: Optional[np.ndarray] = None
        self.img_path: Optional[Path] = None

    def _label_path_for(self, img_path: Path) -> Path:
        return self.labels_dir / (img_path.stem + ".txt")

    def load_current(self) -> None:
        self.img_path = self.images[self.idx]
        self.img = cv2.imread(str(self.img_path))
        if self.img is None:
            raise RuntimeError(f"Không đọc được ảnh: {self.img_path}")
        h, w = self.img.shape[:2]
        self.boxes = read_yolo_labels(self._label_path_for(self.img_path), w=w, h=h)
        self.dragging = False
        self.drag_start = None
        self.drag_end = None

    def save_current(self) -> None:
        assert self.img is not None and self.img_path is not None
        h, w = self.img.shape[:2]
        write_yolo_labels(self._label_path_for(self.img_path), self.boxes, w=w, h=h)
        print(f"✅ Saved: {self._label_path_for(self.img_path)} ({len(self.boxes)} boxes)")

    def next(self) -> None:
        self.idx = min(len(self.images) - 1, self.idx + 1)
        self.load_current()

    def prev(self) -> None:
        self.idx = max(0, self.idx - 1)
        self.load_current()

    def delete_last(self) -> None:
        if self.boxes:
            self.boxes.pop()

    def set_class(self, cid: int) -> None:
        if not self.classes:
            return
        cid = max(0, min(len(self.classes) - 1, cid))
        self.current_class = cid

    def change_class(self, delta: int) -> None:
        if not self.classes:
            return
        self.current_class = (self.current_class + delta) % len(self.classes)

    def on_mouse(self, event, x, y, flags, param) -> None:
        if self.img is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
            self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging and self.drag_start is not None:
                self.dragging = False
                self.drag_end = (x, y)
                x1, y1 = self.drag_start
                x2, y2 = self.drag_end
                # loại bbox quá nhỏ
                if abs(x2 - x1) >= 3 and abs(y2 - y1) >= 3:
                    self.boxes.append(Box(self.current_class, x1, y1, x2, y2))
            self.drag_start = None
            self.drag_end = None

    def render(self) -> np.ndarray:
        assert self.img is not None
        base = self.img
        view = draw_overlay(
            base,
            self.boxes,
            self.classes,
            self.current_class,
            self.idx,
            len(self.images),
            self.help_on,
        )
        if self.dragging and self.drag_start and self.drag_end:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_end
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 165, 255), 2)
        return view


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder ảnh input")
    ap.add_argument("--labels", required=True, help="Folder label output (.txt YOLO)")
    ap.add_argument("--classes", required=True, help="Path classes.txt")
    ap.add_argument("--window", default="YOLO Annotator", help="Tên cửa sổ")
    args = ap.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    classes_path = Path(args.classes)

    if not images_dir.exists():
        print(f"❌ Không tìm thấy images dir: {images_dir}")
        return 1

    classes = load_classes(classes_path)
    imgs = list_images(images_dir)
    if not imgs:
        print(f"❌ Không có ảnh trong: {images_dir}")
        return 1

    ann = Annotator(imgs, labels_dir=labels_dir, classes=classes)
    ann.load_current()

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(args.window, ann.on_mouse)

    while True:
        frame = ann.render()
        cv2.imshow(args.window, frame)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("h"):
            ann.help_on = not ann.help_on
        elif key == ord("s"):
            ann.save_current()
        elif key in (ord("n"), 83):  # 'n' or Right Arrow
            ann.next()
        elif key in (ord("b"), 81):  # 'b' or Left Arrow
            ann.prev()
        elif key == ord("d"):
            ann.delete_last()
        elif key == ord("["):
            ann.change_class(-1)
        elif key == ord("]"):
            ann.change_class(+1)
        elif key == ord("c"):
            print(f"🎯 Current class: {ann.current_class} - {classes[ann.current_class]}")
        # 1..9 map to 0..8
        elif ord("1") <= key <= ord("9"):
            ann.set_class((key - ord("1")))
        elif key == ord("0"):
            ann.set_class(9)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


