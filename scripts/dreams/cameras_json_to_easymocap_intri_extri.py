#!/usr/bin/env python3
"""Convert Dreams cameras.json into EasyMocap/EasyVolcap intri.yml and extri.yml."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write intri.yml and extri.yml from a Dreams cameras.json file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/dreams/take2/frameset/color_frames/000000/cameras.json"),
        help=(
            "Input cameras.json. A frame directory containing cameras.json, or a "
            "color_frames root containing frame subdirectories, is also accepted."
        ),
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/dreams/take2_rearranged"),
        help="Output dataset root where intri.yml and extri.yml will be written.",
    )
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Camera-major image directory under dst. Used to infer camera names when present.",
    )
    parser.add_argument(
        "--name-digits",
        type=int,
        default=2,
        help="Fallback zero-padding for camera names when names cannot be inferred from dst/images.",
    )
    parser.add_argument(
        "--dist-key",
        choices=("dist", "D"),
        default="dist",
        help="Distortion key prefix to write in intri.yml. The sport_2 example uses dist.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=8,
        help="Number of decimal places for matrix values.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing intri.yml or extri.yml.",
    )
    return parser.parse_args()


def resolve_cameras_json(src: Path) -> Path:
    if src.is_file():
        if src.name != "cameras.json":
            raise ValueError(f"Expected a cameras.json file, got: {src}")
        return src

    direct = src / "cameras.json"
    if direct.exists():
        return direct

    candidates = sorted(src.glob("*/cameras.json"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"Could not find cameras.json from: {src}")


def camera_names(dst: Path, images_dir: str, count: int, digits: int) -> list[str]:
    image_root = dst if images_dir == "." else dst / images_dir
    if image_root.exists():
        names = sorted(path.name for path in image_root.iterdir() if path.is_dir())
        if len(names) == count:
            return names
        if names:
            raise RuntimeError(
                f"Found {len(names)} camera folders in {image_root}, but cameras.json has {count} cameras."
            )
    return [f"{idx:0{digits}d}" for idx in range(count)]


def format_number(value: float, precision: int) -> str:
    return f"{float(value):.{precision}f}"


def matrix_block(key: str, value: np.ndarray, precision: int) -> list[str]:
    value = np.asarray(value, dtype=np.float64)
    flat = ", ".join(format_number(v, precision) for v in value.reshape(-1))
    return [
        f"{key}: !!opencv-matrix",
        f"  rows: {value.shape[0]}",
        f"  cols: {value.shape[1]}",
        "  dt: d",
        f"  data: [{flat}]",
    ]


def names_block(names: Iterable[str]) -> list[str]:
    lines = ["names:"]
    lines.extend(f'  - "{name}"' for name in names)
    return lines


def write_text(path: Path, lines: list[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Destination exists, pass --overwrite to replace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_cameras(cameras_json: Path) -> list[dict[str, np.ndarray | int | float]]:
    data = json.loads(cameras_json.read_text(encoding="utf-8"))
    rigs = data.get("rigs")
    if not isinstance(rigs, list) or not rigs:
        raise ValueError(f"No rigs list found in {cameras_json}")

    cameras: list[dict[str, np.ndarray | int | float]] = []
    for idx, rig in enumerate(rigs):
        rig_cameras = rig.get("cameras", [])
        if len(rig_cameras) != 1:
            raise ValueError(f"Rig {idx} should contain exactly one camera, got {len(rig_cameras)}")
        cam = rig_cameras[0]

        R = np.asarray(cam["R"], dtype=np.float64)
        c = np.asarray(cam["c"], dtype=np.float64).reshape(3, 1)
        if R.shape != (3, 3):
            raise ValueError(f"Camera {idx} R should be 3x3, got {R.shape}")

        K = np.array(
            [
                [cam["fx"], 0.0, cam["cx"]],
                [0.0, cam["fy"], cam["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        T = -R @ c
        Rvec = cv2.Rodrigues(R)[0]
        cameras.append(
            {
                "K": K,
                "dist": np.zeros((1, 5), dtype=np.float64),
                "Rvec": Rvec,
                "R": R,
                "T": T,
            }
        )
    return cameras


def main() -> None:
    args = parse_args()
    cameras_json = resolve_cameras_json(args.src)
    cameras = parse_cameras(cameras_json)
    names = camera_names(args.dst, args.images_dir, len(cameras), args.name_digits)

    intri_lines = ["%YAML:1.0", "---", *names_block(names)]
    extri_lines = ["%YAML:1.0", "---", *names_block(names)]

    for name, camera in zip(names, cameras):
        intri_lines.extend(matrix_block(f"K_{name}", camera["K"], args.precision))
        intri_lines.extend(matrix_block(f"{args.dist_key}_{name}", camera["dist"], args.precision))

        extri_lines.extend(matrix_block(f"R_{name}", camera["Rvec"], args.precision))
        extri_lines.extend(matrix_block(f"Rot_{name}", camera["R"], args.precision))
        extri_lines.extend(matrix_block(f"T_{name}", camera["T"], args.precision))

    write_text(args.dst / "intri.yml", intri_lines, args.overwrite)
    write_text(args.dst / "extri.yml", extri_lines, args.overwrite)

    print(f"Read cameras from: {cameras_json}")
    print(f"Wrote {len(cameras)} cameras to: {args.dst / 'intri.yml'}")
    print(f"Wrote {len(cameras)} cameras to: {args.dst / 'extri.yml'}")


if __name__ == "__main__":
    main()
