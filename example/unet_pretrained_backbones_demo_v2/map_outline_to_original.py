"""将 ROI 裁切图上的 outline LabelMe 标注映射回原图坐标。

典型工作流
----------
1. 原图目录下有 LabelMe JSON，其中包含 shape_type="rectangle" 的 ROI 标注
2. 根据 ROI 裁切出子图，裁切图命名为 ``{原图stem}_{roi索引}.{ext}``
3. 在裁切图上用 LabelMe 标注 outline（polygon / polyline 等）
4. 本脚本读取 ROI JSON 和 outline JSON，把 outline 坐标偏移 + 缩放回原图，
   输出一份新的 LabelMe JSON（每张原图一个文件，合并所有 ROI 的 outline）

用法示例::

    python map_outline_to_original.py \
        --images-dir     /path/to/original_images \
        --roi-dir        /path/to/roi_jsons \
        --outline-dir    /path/to/outline_jsons \
        --output-dir     /path/to/output \
        --crop-dir       /path/to/cropped_images        # 可选，用于自动计算缩放
        --roi-label      roi                             # ROI 标注的 label 名（默认取全部 rectangle）
        --crop-pattern   "{stem}_{index}"                # 裁切图命名模式
"""

import json
import re
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_image_size(image_path: Path) -> tuple[int, int]:
    """返回 (width, height)。"""
    from PIL import Image

    with Image.open(image_path) as img:
        return img.size


def find_image_file(directory: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def extract_roi_boxes(
    roi_data: dict,
    roi_label: str | None = None,
) -> list[tuple[float, float, float, float]]:
    """从 LabelMe JSON 中提取 ROI 矩形框，返回 [(x_min, y_min, x_max, y_max), ...]。

    支持 shape_type 为 rectangle（2 点）或 polygon（4 点矩形）。
    如果指定 roi_label，则只取该 label 的 shape。
    """
    boxes = []
    for shape in roi_data.get("shapes", []):
        if roi_label and shape.get("label", "") != roi_label:
            continue

        shape_type = shape.get("shape_type", "")
        points = shape.get("points", [])

        if shape_type == "rectangle" and len(points) >= 2:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        elif shape_type == "polygon" and len(points) >= 3:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))

    if not boxes and not roi_label:
        for shape in roi_data.get("shapes", []):
            points = shape.get("points", [])
            if len(points) >= 2:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


def build_crop_name(stem: str, index: int, pattern: str) -> str:
    """根据模式生成裁切图文件名 stem。"""
    return pattern.replace("{stem}", stem).replace("{index}", str(index))


def parse_crop_name(
    crop_stem: str,
    pattern: str,
) -> tuple[str, int] | None:
    """从裁切图文件名 stem 解析出原图 stem 和 ROI 索引。"""
    regex = re.escape(pattern)
    regex = regex.replace(re.escape("{stem}"), r"(?P<stem>.+)")
    regex = regex.replace(re.escape("{index}"), r"(?P<index>\d+)")
    regex = f"^{regex}$"
    m = re.match(regex, crop_stem)
    if m:
        return m.group("stem"), int(m.group("index"))
    return None


def map_points(
    points: list[list[float]],
    roi_box: tuple[float, float, float, float],
    crop_w: float,
    crop_h: float,
) -> list[list[float]]:
    """将裁切图上的点坐标映射回原图坐标。

    考虑裁切后可能 resize 过，通过 crop_w/crop_h 和 ROI 实际尺寸
    计算缩放因子。
    """
    roi_x_min, roi_y_min, roi_x_max, roi_y_max = roi_box
    roi_w = roi_x_max - roi_x_min
    roi_h = roi_y_max - roi_y_min

    scale_x = roi_w / crop_w if crop_w > 0 else 1.0
    scale_y = roi_h / crop_h if crop_h > 0 else 1.0

    mapped = []
    for px, py in points:
        orig_x = roi_x_min + px * scale_x
        orig_y = roi_y_min + py * scale_y
        mapped.append([round(orig_x, 2), round(orig_y, 2)])
    return mapped


def process(
    images_dir: Path,
    roi_dir: Path,
    outline_dir: Path,
    output_dir: Path,
    crop_dir: Path | None,
    roi_label: str | None,
    crop_pattern: str,
    include_roi_shapes: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_jsons = sorted(roi_dir.glob("*.json"))
    if not roi_jsons:
        print(f"No ROI JSON files found in {roi_dir}")
        return

    total_mapped = 0
    total_images = 0

    for roi_json_path in roi_jsons:
        stem = roi_json_path.stem
        roi_data = load_json(roi_json_path)
        roi_boxes = extract_roi_boxes(roi_data, roi_label)

        if not roi_boxes:
            print(f"  [{stem}] No ROI boxes found, skipping")
            continue

        orig_image = find_image_file(images_dir, stem)
        if orig_image is None:
            print(f"  [{stem}] Original image not found in {images_dir}, skipping")
            continue

        orig_w, orig_h = get_image_size(orig_image)

        output_shapes: list[dict] = []

        if include_roi_shapes:
            for shape in roi_data.get("shapes", []):
                output_shapes.append(dict(shape))

        for roi_idx, roi_box in enumerate(roi_boxes, start=1):
            crop_stem = build_crop_name(stem, roi_idx, crop_pattern)
            outline_json_path = outline_dir / f"{crop_stem}.json"

            if not outline_json_path.is_file():
                print(f"  [{stem}] Outline JSON not found: {outline_json_path.name}, skipping ROI #{roi_idx}")
                continue

            outline_data = load_json(outline_json_path)

            crop_w = float(outline_data.get("imageWidth", 0))
            crop_h = float(outline_data.get("imageHeight", 0))

            if (crop_w <= 0 or crop_h <= 0) and crop_dir is not None:
                crop_image = find_image_file(crop_dir, crop_stem)
                if crop_image is not None:
                    cw, ch = get_image_size(crop_image)
                    crop_w, crop_h = float(cw), float(ch)

            if crop_w <= 0 or crop_h <= 0:
                roi_w = roi_box[2] - roi_box[0]
                roi_h = roi_box[3] - roi_box[1]
                crop_w, crop_h = roi_w, roi_h
                print(f"  [{stem}] ROI #{roi_idx}: crop size unknown, assuming same as ROI ({roi_w:.0f}x{roi_h:.0f})")

            outline_shapes = outline_data.get("shapes", [])
            mapped_count = 0

            for shape in outline_shapes:
                points = shape.get("points", [])
                if not points:
                    continue

                mapped_points = map_points(points, roi_box, crop_w, crop_h)

                new_shape = {
                    "label": shape.get("label", "outline"),
                    "points": mapped_points,
                    "group_id": shape.get("group_id"),
                    "description": shape.get("description", ""),
                    "shape_type": shape.get("shape_type", "polygon"),
                    "flags": shape.get("flags", {}),
                    "mask": shape.get("mask"),
                }
                output_shapes.append(new_shape)
                mapped_count += 1

            total_mapped += mapped_count
            print(f"  [{stem}] ROI #{roi_idx}: mapped {mapped_count} outline shape(s)")

        if not output_shapes:
            continue

        output_data = {
            "version": roi_data.get("version", "5.4.1"),
            "flags": roi_data.get("flags", {}),
            "shapes": output_shapes,
            "imagePath": orig_image.name,
            "imageData": None,
            "imageHeight": orig_h,
            "imageWidth": orig_w,
        }

        output_path = output_dir / f"{stem}.json"
        save_json(output_path, output_data)
        total_images += 1
        print(f"  [{stem}] Saved → {output_path.name}")

    print(f"\nDone: {total_images} image(s), {total_mapped} outline shape(s) mapped")


if __name__ == "__main__":
    # ===================== 在这里修改参数 =====================
    IMAGES_DIR = r"/path/to/original_images"       # 原图目录
    ROI_DIR = r"/path/to/roi_jsons"                # ROI LabelMe JSON 目录（与原图同名 .json）
    OUTLINE_DIR = r"/path/to/outline_jsons"        # 裁切图上 outline LabelMe JSON 目录
    OUTPUT_DIR = r"/path/to/output"                # 输出目录，保存映射后的 LabelMe JSON
    CROP_DIR = None                                # 裁切图目录（可选，用于读取裁切图实际尺寸以计算缩放）
    ROI_LABEL = None                               # 只取该 label 的 ROI shape（默认 None 取全部 rectangle）
    CROP_PATTERN = "{stem}_{index}"                # 裁切图命名模式（索引从 1 开始：stem_1, stem_2, ...）
    INCLUDE_ROI = False                            # 在输出 JSON 中保留原始 ROI 标注
    # =========================================================

    process(
        images_dir=Path(IMAGES_DIR),
        roi_dir=Path(ROI_DIR),
        outline_dir=Path(OUTLINE_DIR),
        output_dir=Path(OUTPUT_DIR),
        crop_dir=Path(CROP_DIR) if CROP_DIR else None,
        roi_label=ROI_LABEL,
        crop_pattern=CROP_PATTERN,
        include_roi_shapes=INCLUDE_ROI,
    )
