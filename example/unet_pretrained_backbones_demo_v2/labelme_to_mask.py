"""Render LabelMe JSON annotations into grayscale segmentation masks.

Each LabelMe ``.json`` (polygon / rectangle / linestrip / points) is rasterized
into a single-channel PNG whose pixel value encodes the class:

- binary mode (default): every labeled shape is filled with 255 (foreground),
  matching ``SegmentationTxtDataset`` with ``num_classes == 1`` (mask > 0 -> 1).
- multi-class mode (``--class-names a b c``): class ``a`` -> 1, ``b`` -> 2, ...
  (0 = background). Feed the same order to training via ``--mask-values`` so the
  raw pixel values map back to class indices.

Two layouts are supported:

- flat:      ``--json-dir <dir> --output-dir <dir>``  (one mask per json by stem)
- recursive: ``--root <root> [--json-subdir labels] [--out-subdir masks]`` — for
  every ``.../<json-subdir>/*.json`` a mask is written to the sibling
  ``.../<out-subdir>/<stem>.png``. This mirrors the ``类别/日期/{images,labels}``
  layout; afterwards run ``prepare_splits.py --recursive --masks-subdir masks``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common import IMAGE_EXTENSIONS

# shape_types whose points enclose a filled region
POLYGON_LIKE = {"polygon", "linestrip", "points", None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LabelMe json to grayscale masks.")
    parser.add_argument("--json-dir", type=str, default=None, help="平铺模式：LabelMe json 文件夹。")
    parser.add_argument("--images-dir", type=str, default=None, help="可选：图片文件夹，用于在 json 缺 imageWidth/Height 时取尺寸。")
    parser.add_argument("--output-dir", type=str, default=None, help="平铺模式：mask 输出文件夹。")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="多类：按顺序把标签映射到像素值 1..N(0=背景)。不给则二值(所有形状=255)。",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归分层模式：把 --root 下每个 <json-subdir>/*.json 渲染到同级 <out-subdir>/<stem>.png。",
    )
    parser.add_argument("--root", type=str, default=None, help="递归模式的数据集根。")
    parser.add_argument("--json-subdir", type=str, default="labels", help="递归模式 json 子目录名(默认 labels)。")
    parser.add_argument("--out-subdir", type=str, default="masks", help="递归模式 mask 输出子目录名(默认 masks)。")
    parser.add_argument("--images-subdir", type=str, default="images", help="递归模式图片子目录名(用于取尺寸，默认 images)。")
    return parser.parse_args()


def build_class_to_value(class_names):
    """标签 -> 像素值。多类为 1..N；二值(class_names 为空)时统一用 255。"""
    if class_names:
        return {name: index + 1 for index, name in enumerate(class_names)}
    return None


def find_image_for(json_path: Path, image_dirs):
    """按 stem 在候选目录里找同名图片(用于取尺寸)。"""
    for image_dir in image_dirs:
        if image_dir is None:
            continue
        for suffix in IMAGE_EXTENSIONS:
            candidate = image_dir / f"{json_path.stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def resolve_size(data: dict, json_path: Path, image_dirs):
    """取 (width, height)：优先 json 里的 imageWidth/Height，否则打开配对图片。"""
    width = data.get("imageWidth")
    height = data.get("imageHeight")
    if width and height:
        return int(width), int(height)
    image_path = find_image_for(json_path, image_dirs)
    if image_path is not None:
        with Image.open(image_path) as image:
            return image.size
    return None


def render_mask(data: dict, size, class_to_value):
    """把 json 的 shapes 画到一张 L 模式 mask 上。返回 (mask, dropped, unknown_labels)。"""
    width, height = size
    mask = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(mask)
    dropped = 0
    unknown = set()
    for shape in data.get("shapes", []):
        label = shape.get("label")
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "polygon")

        if class_to_value is None:
            value = 255
        elif label in class_to_value:
            value = class_to_value[label]
        else:
            unknown.add(label)
            dropped += 1
            continue

        tuples = [(float(x), float(y)) for x, y in points]
        if shape_type == "rectangle" and len(tuples) == 2:
            (x0, y0), (x1, y1) = tuples
            drawer.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=value)
        elif shape_type == "circle" and len(tuples) == 2:
            (cx, cy), (px, py) = tuples
            radius = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            drawer.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=value)
        elif shape_type in POLYGON_LIKE and len(tuples) >= 3:
            drawer.polygon(tuples, fill=value)
        else:
            dropped += 1
    return mask, dropped, unknown


def iter_jobs(args):
    """产出 (json_path, output_mask_path, image_dirs) 三元组。"""
    if args.recursive:
        if not args.root:
            raise ValueError("--recursive 需要 --root。")
        root = Path(args.root)
        json_dirs = [p for p in root.rglob("*") if p.is_dir() and p.name == args.json_subdir]
        for json_dir in sorted(json_dirs):
            out_dir = json_dir.parent / args.out_subdir
            image_dirs = [json_dir.parent / args.images_subdir, json_dir]
            for json_path in sorted(json_dir.glob("*.json")):
                yield json_path, out_dir / f"{json_path.stem}.png", image_dirs
    else:
        if not args.json_dir or not args.output_dir:
            raise ValueError("平铺模式需要 --json-dir 和 --output-dir。")
        json_dir = Path(args.json_dir)
        out_dir = Path(args.output_dir)
        image_dirs = [Path(args.images_dir) if args.images_dir else None, json_dir]
        for json_path in sorted(json_dir.glob("*.json")):
            yield json_path, out_dir / f"{json_path.stem}.png", image_dirs


def main() -> None:
    args = parse_args()
    class_to_value = build_class_to_value(args.class_names)

    written = 0
    skipped = 0
    total_dropped = 0
    all_unknown = set()
    for json_path, mask_path, image_dirs in iter_jobs(args):
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        size = resolve_size(data, json_path, image_dirs)
        if size is None:
            print(f"Skip {json_path}: 无 imageWidth/Height 且找不到配对图片。")
            skipped += 1
            continue
        mask, dropped, unknown = render_mask(data, size, class_to_value)
        total_dropped += dropped
        all_unknown |= unknown
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_path)
        written += 1

    if class_to_value is not None:
        mapping = ", ".join(f"{name}={value}" for name, value in class_to_value.items())
        print(f"Class pixel values: {mapping} (0=background). 训练时用 --mask-values 0 {' '.join(str(v) for v in class_to_value.values())} 对齐。")
    if all_unknown:
        print(f"Warning: 未在 --class-names 中的标签被忽略: {sorted(all_unknown)}")
    if total_dropped:
        print(f"Warning: {total_dropped} 个形状点数不足/类型不支持被跳过。")
    print(f"Done. masks written={written}, skipped_json={skipped}")


if __name__ == "__main__":
    main()
