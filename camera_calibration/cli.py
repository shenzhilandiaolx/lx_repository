from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .calibrator import CameraCalibrator, CheckerboardSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="棋盘格相机标定工具（无 OpenCV）")
    parser.add_argument("image_dir", type=Path, help="棋盘格图片目录")
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--square-size", type=float, required=True)
    parser.add_argument("--output", type=Path, default=Path("calibration_result.npz"))
    return parser


def _load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64)


def main() -> None:
    args = build_parser().parse_args()
    image_paths = sorted([p for p in args.image_dir.glob("*") if p.is_file()])
    if len(image_paths) < 3:
        raise RuntimeError("至少需要 3 张棋盘格图片")

    board = CheckerboardSpec(args.cols, args.rows, args.square_size)
    calibrator = CameraCalibrator(board)
    gray_images = [_load_gray(p) for p in image_paths]

    result = calibrator.calibrate_from_images(gray_images)
    calibrator.save(result, args.output)

    print(f"标定完成: RMS={result.rms:.6f}")
    print("K=")
    print(result.camera_matrix)
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
