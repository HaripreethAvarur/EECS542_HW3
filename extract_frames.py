#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, output_dir: Path, prefix: str, zero_pad: int, image_ext: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        filename = f"{prefix}_{idx:0{zero_pad}d}.{image_ext}"
        cv2.imwrite(str(output_dir / filename), frame)
        idx += 1

    cap.release()

    print(f"Video FPS: {fps:.3f} | Reported frames: {total_frames} | Extracted: {idx}")
    return idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all frames from a video into a folder (original FPS count)."
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for frames (default: <video_stem>_frames next to video)",
    )
    parser.add_argument("--prefix", default="frame", help="Output filename prefix")
    parser.add_argument("--zero_pad", type=int, default=6, help="Zero-pad width for indices")
    parser.add_argument(
        "--ext",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Image format for saved frames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).resolve()
    if args.out_dir:
        output_dir = Path(args.out_dir).resolve()
    else:
        output_dir = video_path.parent / f"{video_path.stem}_frames"

    num = extract_frames(video_path, output_dir, args.prefix, args.zero_pad, args.ext)
    print(f"Saved {num} frames to: {output_dir}")


if __name__ == "__main__":
    main()


