"""
Trích xuất frame từ video và lưu ra folder ảnh để annotation.

Output structure (mặc định):
  <out_dir>/
    images/
      frame_000001.jpg
      frame_000151.jpg
    frames_manifest.jsonl   (mỗi dòng 1 JSON)

Ví dụ:
  python tools/extract_frames_from_video.py --video C:\\path\\video.mp4 --out dataset_raw --every 30
  python tools/extract_frames_from_video.py --video video.mp4 --out dataset_raw --frames 10,20,21,300
  python tools/extract_frames_from_video.py --video video.mp4 --out dataset_raw --start 100 --end 500 --every 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Set

import cv2


def _parse_frames_arg(frames: Optional[str]) -> Optional[Set[int]]:
    if not frames:
        return None
    # chấp nhận: "1,2,3" hoặc "1 2 3"
    raw = frames.replace(" ", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: Set[int] = set()
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            if b_i < a_i:
                a_i, b_i = b_i, a_i
            out.update(range(a_i, b_i + 1))
        else:
            out.add(int(p))
    return out


def _iter_target_frames(
    total_frames: int,
    frames_set: Optional[Set[int]],
    start: int,
    end: Optional[int],
    every: int,
    max_frames: Optional[int],
) -> List[int]:
    if start < 0:
        start = 0
    if end is None:
        end = total_frames - 1 if total_frames > 0 else None
    if end is not None and end < start:
        start, end = end, start

    targets: List[int] = []
    if frames_set is not None:
        # lọc theo range
        for f in sorted(frames_set):
            if f < start:
                continue
            if end is not None and f > end:
                continue
            targets.append(f)
    else:
        if every <= 0:
            every = 1
        if end is None:
            # không biết total_frames => cứ generate đến khi cap.read() fail, nhưng script này cần seek
            # fallback: chỉ lấy start
            targets = [start]
        else:
            targets = list(range(start, end + 1, every))

    if max_frames is not None and max_frames > 0:
        targets = targets[: max_frames]
    return targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Đường dẫn video")
    ap.add_argument("--out", required=True, help="Thư mục output")
    ap.add_argument("--prefix", default="frame_", help="Prefix tên file ảnh")
    ap.add_argument("--ext", default="jpg", choices=["jpg", "png"], help="Định dạng ảnh output")
    ap.add_argument("--quality", type=int, default=90, help="Chất lượng JPEG (1-100), chỉ áp dụng khi ext=jpg")
    ap.add_argument("--frames", default=None, help="Danh sách frame index: '10,20,21' hoặc range '100-200'")
    ap.add_argument("--start", type=int, default=0, help="Frame bắt đầu (khi không dùng --frames)")
    ap.add_argument("--end", type=int, default=None, help="Frame kết thúc (khi không dùng --frames)")
    ap.add_argument("--every", type=int, default=30, help="Lấy mỗi N frame (khi không dùng --frames)")
    ap.add_argument("--max_frames", type=int, default=None, help="Giới hạn số frame trích xuất")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Không tìm thấy video: {video_path}")
        return 1

    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "frames_manifest.jsonl"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames_set = _parse_frames_arg(args.frames)
    targets = _iter_target_frames(
        total_frames=total_frames,
        frames_set=frames_set,
        start=args.start,
        end=args.end,
        every=args.every,
        max_frames=args.max_frames,
    )
    if not targets:
        print("⚠️ Không có frame nào để trích xuất (check --frames/--start/--end).")
        return 1

    jpeg_params = []
    if args.ext == "jpg":
        q = max(1, min(100, int(args.quality)))
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), q]

    extracted = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for frame_idx in targets:
            # seek theo frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            filename = f"{args.prefix}{frame_idx:06d}.{args.ext}"
            out_path = images_dir / filename
            if args.ext == "jpg":
                cv2.imwrite(str(out_path), frame, jpeg_params)
            else:
                cv2.imwrite(str(out_path), frame)

            ts_sec = (frame_idx / fps) if fps and fps > 0 else None
            rec = {
                "video": str(video_path),
                "frame_index": frame_idx,
                "timestamp_sec": ts_sec,
                "image_file": str(out_path),
                "fps": fps,
                "width": w,
                "height": h,
            }
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            extracted += 1

    cap.release()
    print(f"✅ Đã trích xuất {extracted} frame → {images_dir}")
    print(f"📝 Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


