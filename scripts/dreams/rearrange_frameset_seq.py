#!/usr/bin/env python3
"""Rearrange frame-major Dreams exports into EasyVolcap camera-major layout.

Input layout:
    color_frames/000000/00.png
    color_frames/000000/01.png
    color_frames/000001/00.png

Default output layout:
    data/dreams/seq0/images/00/000000.png
    data/dreams/seq0/images/01/000000.png
    data/dreams/seq0/images/00/000001.png

If the input contains RGBA foreground cutouts instead of raw images, pass
--rgba-foreground-only to write RGB composites to images/ and alpha masks to masks/.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, TypeVar


CAMERA_FILE_RE = re.compile(r"^(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rearrange frameset_seq/color_frames from frame-major to camera-major layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/example_raw_data/frameset_seq/color_frames"),
        help="Input frame-major image directory containing frame folders.",
    )
    parser.add_argument(
        "--mask-src",
        type=Path,
        default=None,
        help="Optional input frame-major mask directory. When set, masks are rearranged into masks-dir.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/dreams/seq0"),
        help="Output dataset root.",
    )
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Image directory under dst. Use '.' to put camera folders directly under dst.",
    )
    parser.add_argument(
        "--masks-dir",
        default="masks",
        help="Mask directory under dst.",
    )
    parser.add_argument(
        "--rgba-foreground-only",
        action="store_true",
        help="Treat src as RGBA foreground cutouts: save RGB composites to images-dir and alpha masks to masks-dir.",
    )
    parser.add_argument(
        "--background",
        default="0,0,0",
        help="RGB background color for --rgba-foreground-only composites.",
    )
    parser.add_argument(
        "--method",
        choices=("copy", "symlink", "hardlink"),
        default="copy",
        help="How to place files in the output tree. RGBA extraction always writes new PNG files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination files.",
    )
    parser.add_argument(
        "--renumber",
        action="store_true",
        help="Rename frames to contiguous indices in sorted order instead of preserving source frame names.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="Number of digits for output frame names when --renumber is used.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def progress(items: Iterable[T], total: int, desc: str, enabled: bool) -> Iterable[T]:
    if not enabled:
        return items
    try:
        from tqdm import tqdm
    except ImportError:
        print(f"{desc}: processing {total} files")
        return items
    return tqdm(items, total=total, desc=desc, unit="file")


def list_frame_dirs(src: Path) -> list[Path]:
    if not src.exists():
        raise FileNotFoundError(f"Input directory does not exist: {src}")
    frame_dirs = [p for p in src.iterdir() if p.is_dir()]
    return sorted(frame_dirs, key=lambda p: p.name)


def list_camera_images(frame_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(frame_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        match = CAMERA_FILE_RE.match(path.name)
        if not match:
            continue
        cam_id = match.group(1)
        images[cam_id] = path
    return images


def place_file(src: Path, dst: Path, method: str, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Destination exists, pass --overwrite to replace: {dst}")
        dst.unlink()
    if method == "copy":
        shutil.copy2(src, dst)
    elif method == "symlink":
        os.symlink(src.resolve(), dst)
    elif method == "hardlink":
        os.link(src, dst)
    elif method == "move":
        shutil.move(src, dst)
    else:
        raise ValueError(f"Unsupported method: {method}")


def ensure_can_write(dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Destination exists, pass --overwrite to replace: {dst}")
        dst.unlink()


def parse_background(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError("--background must be formatted like '0,0,0'")
    color = tuple(int(p) for p in parts)
    if any(c < 0 or c > 255 for c in color):
        raise ValueError("--background values must be in [0, 255]")
    return color  # type: ignore[return-value]


def save_rgba_as_image_and_mask(
    src: Path,
    image_dst: Path,
    mask_dst: Path,
    background: tuple[int, int, int],
    overwrite: bool,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for --rgba-foreground-only. Install it with `pip install pillow`.") from exc

    with Image.open(src) as im:
        rgba = im.convert("RGBA")
        alpha = rgba.getchannel("A")
        rgb = Image.new("RGB", rgba.size, background)
        rgb.paste(rgba.convert("RGB"), mask=alpha)

    ensure_can_write(image_dst, overwrite)
    rgb.save(image_dst)

    ensure_can_write(mask_dst, overwrite)
    alpha.save(mask_dst)


def rearrange_tree(
    src: Path,
    output_root: Path,
    args: argparse.Namespace,
    label: str,
) -> tuple[list[str], int, int, list[str]]:
    frame_dirs = list_frame_dirs(src)
    if not frame_dirs:
        raise RuntimeError(f"No frame folders found in {src}")

    expected_cameras: list[str] | None = None
    placed = 0
    ignored = 0
    inconsistent: list[str] = []
    jobs: list[tuple[Path, Path]] = []

    for out_idx, frame_dir in enumerate(frame_dirs):
        frame_images = list_camera_images(frame_dir)
        ignored += sum(1 for p in frame_dir.iterdir() if p.is_file() and not CAMERA_FILE_RE.match(p.name))
        camera_ids = sorted(frame_images)
        if expected_cameras is None:
            expected_cameras = camera_ids
        elif camera_ids != expected_cameras:
            inconsistent.append(frame_dir.name)

        out_frame_name = f"{out_idx:0{args.digits}d}" if args.renumber else frame_dir.name
        for cam_id, src_path in frame_images.items():
            dst_path = output_root / cam_id / f"{out_frame_name}{src_path.suffix.lower()}"
            jobs.append((src_path, dst_path))

    for src_path, dst_path in progress(jobs, len(jobs), label, not args.no_progress):
        place_file(src_path, dst_path, args.method, args.overwrite, args.dry_run)
        placed += 1

    print(f"{label} source: {src}")
    print(f"{label} destination: {output_root}")
    return expected_cameras or [], len(frame_dirs), placed, inconsistent + [f"ignored:{ignored}"]


def main() -> None:
    args = parse_args()
    if args.rgba_foreground_only and args.mask_src is not None:
        raise ValueError("--rgba-foreground-only and --mask-src are mutually exclusive")

    frame_dirs = list_frame_dirs(args.src)
    if not frame_dirs:
        raise RuntimeError(f"No frame folders found in {args.src}")

    images_root = args.dst if args.images_dir in ("", ".") else args.dst / args.images_dir
    masks_root = args.dst / args.masks_dir
    expected_cameras: list[str] | None = None
    placed = 0
    masks_placed = 0
    ignored = 0
    inconsistent: list[str] = []
    background = parse_background(args.background)
    image_jobs: list[tuple[Path, Path, Path | None]] = []

    for out_idx, frame_dir in enumerate(frame_dirs):
        frame_images = list_camera_images(frame_dir)
        ignored += sum(1 for p in frame_dir.iterdir() if p.is_file() and not CAMERA_FILE_RE.match(p.name))
        camera_ids = sorted(frame_images)
        if expected_cameras is None:
            expected_cameras = camera_ids
        elif camera_ids != expected_cameras:
            inconsistent.append(frame_dir.name)

        out_frame_name = f"{out_idx:0{args.digits}d}" if args.renumber else frame_dir.name
        for cam_id, src_path in frame_images.items():
            dst_path = images_root / cam_id / f"{out_frame_name}{src_path.suffix.lower()}"
            if args.rgba_foreground_only:
                image_dst = images_root / cam_id / f"{out_frame_name}.png"
                mask_dst = masks_root / cam_id / f"{out_frame_name}.png"
                image_jobs.append((src_path, image_dst, mask_dst))
            else:
                image_jobs.append((src_path, dst_path, None))

    for src_path, image_dst, mask_dst in progress(image_jobs, len(image_jobs), "images", not args.no_progress):
        if mask_dst is not None:
            save_rgba_as_image_and_mask(
                src_path,
                image_dst,
                mask_dst,
                background,
                args.overwrite,
                args.dry_run,
            )
            placed += 1
            masks_placed += 1
        else:
            place_file(src_path, image_dst, args.method, args.overwrite, args.dry_run)
            placed += 1

    if args.mask_src is not None:
        mask_cameras, mask_frames, masks_placed, mask_inconsistent = rearrange_tree(args.mask_src, masks_root, args, "mask")
        if expected_cameras is not None and mask_cameras != expected_cameras:
            inconsistent.append("mask source camera set differs from image source")
        if mask_frames != len(frame_dirs):
            inconsistent.append(f"mask source has {mask_frames} frame folders, image source has {len(frame_dirs)}")
        for entry in mask_inconsistent:
            if entry.startswith("ignored:"):
                ignored += int(entry.split(":", 1)[1])
            else:
                inconsistent.append(f"mask:{entry}")

    print(f"source: {args.src}")
    print(f"destination images: {images_root}")
    if args.rgba_foreground_only or args.mask_src is not None:
        print(f"destination masks: {masks_root}")
    print(f"frames: {len(frame_dirs)}")
    print(f"cameras: {len(expected_cameras or [])} ({', '.join(expected_cameras or [])})")
    print(f"image files {'would be placed' if args.dry_run else 'placed'}: {placed}")
    if args.rgba_foreground_only or args.mask_src is not None:
        print(f"mask files {'would be placed' if args.dry_run else 'placed'}: {masks_placed}")
    if ignored:
        print(f"ignored non-camera files: {ignored}")
    if inconsistent:
        print("warning: frames with camera set different from first frame:")
        for name in inconsistent[:20]:
            print(f"  {name}")
        if len(inconsistent) > 20:
            print(f"  ... {len(inconsistent) - 20} more")


if __name__ == "__main__":
    main()
