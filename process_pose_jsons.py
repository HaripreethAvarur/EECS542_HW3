#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def list_json_files(input_dir: Path) -> List[Path]:
    """Return sorted list of .json files in the directory."""
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
    files.sort()
    return files


def load_keypoints_first_person(json_path: Path) -> np.ndarray:
    """Load first person's pose_keypoints_2d as (3, num_joints) float32 array.

    If no person is found, returns an empty (3, 0) array.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    people = data.get("people", [])
    if not people:
        return np.zeros((3, 0), dtype=np.float32)

    key_list = people[0]["pose_keypoints_2d"]
    arr = np.asarray(key_list, dtype=np.float32).reshape(-1, 3).T
    return arr


def process_directory(input_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = list_json_files(input_dir)
    count = 0
    for idx, json_path in enumerate(json_files):
        out_path = output_dir / f"{idx:04d}_joints.npy"
        arr = load_keypoints_first_person(json_path)
        # Save regardless of shape (empty arrays become (3, 0))
        np.save(out_path, arr)
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process a directory of OpenPose-style JSON files and save one "
            "(3, num_joints) numpy array per file as <basename>_joints.npy."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing JSON files (one per frame).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save outputs. Defaults to input_dir if not set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir

    num_processed = process_directory(input_dir, output_dir)
    print(
        f"Processed {num_processed} JSON files. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()


