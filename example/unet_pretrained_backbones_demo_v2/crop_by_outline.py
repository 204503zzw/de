"""Crop image + bubble labels the same way X-AnyLabeling's "Save Cropped Image" does.

裁切几何与命名完全对齐 X-AnyLabeling (anylabeling/views/labeling/utils/crop.py)：

- bbox 取 ``cv2.boundingRect(np.int32(points))``（点先截断为整数），再按图像边界裁剪；
- 只处理 ``rectangle`` / ``polygon`` / ``rotation`` 且点数 >= 3 的形状；
- ``min_width`` / ``min_height`` 过滤在 clamp 之前、对 boundingRect 的 w/h 判断；
- 输出目录按标签分子目录：``<output_dir>/<label>/<图名>_<序号>-<shape_type>.jpg``，
  序号为每张图内该标签的计数（从 1 开始），默认存 jpg。

相对 X-AnyLabeling 额外保留的能力（软件本身不做）：同步裁剪目标标签（默认 "bubble"），
用 shapely 做几何交集，跨边界的气泡按裁切框正确切分，不会整条丢失，也不会像逐点 clamp 那样变形。

依赖: pip install shapely opencv-python numpy
"""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.validation import make_valid

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CROPPABLE_SHAPE_TYPES = ("rectangle", "polygon", "rotation")


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, image, params=None):
    path = Path(path)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def find_image(image_name, json_path, image_dir):
    candidates = []
    if image_name:
        candidates.append(json_path.parent / image_name)
        if image_dir:
            candidates.append(image_dir / image_name)
            candidates.append(image_dir / Path(image_name).name)
    for suffix in IMAGE_SUFFIXES:
        candidates.append(json_path.with_suffix(suffix))
        if image_dir:
            candidates.append(image_dir / f"{json_path.stem}{suffix}")
    for c in candidates:
        if c and c.exists():
            return c
    return None


def anylabeling_bbox(points, img_w, img_h, min_width=0, min_height=0):
    """X-AnyLabeling 的裁切框：int32 点集的 boundingRect，再 clamp 到图像内。

    返回 (x1, y1, x2, y2)；不满足最小宽高或裁切框无效时返回 None。
    """
    pts = np.array(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 3:
        return None
    pts = pts.astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    if w < min_width or h < min_height:
        return None
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def _rings_from_geom(geom):
    rings = []
    gt = geom.geom_type
    if gt == "Polygon":
        polys = [geom]
    elif gt == "MultiPolygon":
        polys = list(geom.geoms)
    elif gt == "GeometryCollection":
        polys = []
        for g in geom.geoms:
            if g.geom_type == "Polygon":
                polys.append(g)
            elif g.geom_type == "MultiPolygon":
                polys.extend(list(g.geoms))
    else:
        return rings
    for p in polys:
        if not p.is_empty and p.area > 0:
            rings.append([[float(x), float(y)] for x, y in list(p.exterior.coords)[:-1]])
    return rings


def clip_shape_to_crop(local_pts, crop_w, crop_h, shape_type, min_area=0.5):
    """Clip a shape (already shifted into crop-local coords) to the crop rect.

    Returns list of point-rings (one per resulting piece); [] -> drop the shape.
    """
    rect = box(0, 0, crop_w, crop_h)
    if shape_type in ("polygon", "rectangle", "rotation") and len(local_pts) >= 3:
        poly = Polygon(local_pts)
        if not poly.is_valid:
            poly = make_valid(poly)
        inter = poly.intersection(rect)
        if inter.is_empty or inter.area < min_area:
            return []
        return _rings_from_geom(inter)
    if len(local_pts) >= 2:
        line = LineString(local_pts).intersection(rect)
        if line.is_empty:
            return []
        rings = []
        geoms = list(line.geoms) if line.geom_type.startswith("Multi") else [line]
        for g in geoms:
            if g.geom_type == "LineString" and len(g.coords) >= 2:
                rings.append([[float(x), float(y)] for x, y in g.coords])
        return rings
    return []


def crop_by_outline(
    roi_json_dir,
    output_dir,
    target_json_dir=None,
    image_dir=None,
    roi_label="outline",
    target_labels=("bubble",),
    image_ext=".jpg",
    jpeg_quality=95,
    min_area=0.5,
    min_width=0,
    min_height=0,
    save_labels=True,
):
    """按外轮廓多边形 bbox 裁切图片（X-AnyLabeling 方式），并同步裁剪目标(气泡)标签。

    roi_label     : 外轮廓多边形的标签名（裁切依据）
    target_labels : 需要同步保留/裁剪的标签名列表；None -> 除 roi_label 外全部保留
    min_width/min_height : 与 X-AnyLabeling 一致的最小宽高过滤
    save_labels   : 是否在裁切图旁写出裁剪后的 LabelMe JSON
    """
    roi_json_dir = Path(roi_json_dir)
    output_dir = Path(output_dir)
    target_json_dir = Path(target_json_dir) if target_json_dir else None
    image_dir = Path(image_dir) if image_dir else None
    separate_dirs = target_json_dir is not None

    output_dir.mkdir(parents=True, exist_ok=True)

    target_set = set(target_labels) if target_labels else None
    total_crops = 0
    total_shapes = 0
    skipped_files = 0

    target_json_map = {}
    if separate_dirs:
        for p in target_json_dir.glob("*.json"):
            target_json_map[p.stem] = p

    for roi_json_path in sorted(roi_json_dir.glob("*.json")):
        with open(roi_json_path, "r", encoding="utf-8") as f:
            roi_data = json.load(f)

        image_name = roi_data.get("imagePath") or f"{roi_json_path.stem}.png"
        image_path = find_image(image_name, roi_json_path, image_dir)
        if image_path is None and separate_dirs and image_dir is None:
            tgt_path = target_json_map.get(roi_json_path.stem)
            if tgt_path:
                image_path = find_image(image_name, tgt_path, None)
        if image_path is None:
            print(f"Skip {roi_json_path.name}: image not found")
            skipped_files += 1
            continue

        image = imread_unicode(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skip {roi_json_path.name}: failed to read image {image_path}")
            skipped_files += 1
            continue
        img_h, img_w = image.shape[:2]

        # --- 外轮廓多边形（裁切依据）---
        roi_shapes = [
            s
            for s in roi_data.get("shapes", [])
            if s.get("label", "") == roi_label
            and s.get("shape_type", "polygon") in CROPPABLE_SHAPE_TYPES
        ]
        if not roi_shapes:
            print(f"Skip {roi_json_path.name}: no croppable '{roi_label}' shapes")
            skipped_files += 1
            continue

        # --- 目标(气泡)标签所在的 JSON ---
        if separate_dirs:
            tgt_json_path = target_json_map.get(roi_json_path.stem)
            if tgt_json_path is None:
                print(f"Skip {roi_json_path.name}: no matching target JSON")
                skipped_files += 1
                continue
            with open(tgt_json_path, "r", encoding="utf-8") as f:
                tgt_data = json.load(f)
            all_tgt_shapes = tgt_data.get("shapes", [])
        else:
            tgt_data = roi_data
            all_tgt_shapes = roi_data.get("shapes", [])

        other_shapes = []
        for shape in all_tgt_shapes:
            label = shape.get("label", "")
            if label == roi_label:
                continue
            if target_set is None or label in target_set:
                other_shapes.append(shape)

        stem = image_path.stem
        # X-AnyLabeling: 每张图内按标签计数，序号从 1 开始
        label_to_count = {}

        for roi_shape in roi_shapes:
            roi_pts = roi_shape.get("points", [])
            shape_type = roi_shape.get("shape_type", "polygon")

            bbox = anylabeling_bbox(
                roi_pts, img_w, img_h, min_width=min_width, min_height=min_height
            )
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            crop_w, crop_h = x2 - x1, y2 - y1

            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            new_shapes = []
            for shape in other_shapes:
                pts = shape.get("points", [])
                if len(pts) < 2:
                    continue
                # 平移到裁切局部坐标
                local_pts = [[px - x1, py - y1] for px, py in pts]
                rings = clip_shape_to_crop(
                    local_pts,
                    crop_w,
                    crop_h,
                    shape.get("shape_type", "polygon"),
                    min_area=min_area,
                )
                for ring in rings:
                    new_shape = copy.deepcopy(shape)
                    new_shape["points"] = ring
                    new_shapes.append(new_shape)

            # 输出路径与命名对齐 X-AnyLabeling
            label_dir = output_dir / roi_label
            label_dir.mkdir(parents=True, exist_ok=True)
            label_to_count[roi_label] = label_to_count.get(roi_label, 0) + 1
            crop_name = f"{stem}_{label_to_count[roi_label]}-{shape_type}"

            img_out = label_dir / f"{crop_name}{image_ext}"
            params = []
            if image_ext.lower() in {".jpg", ".jpeg"}:
                params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            imwrite_unicode(img_out, cropped, params)

            if save_labels:
                new_data = {
                    "version": tgt_data.get("version", "5.0.1"),
                    "flags": tgt_data.get("flags", {}),
                    "shapes": new_shapes,
                    "imagePath": f"{crop_name}{image_ext}",
                    "imageData": None,
                    "imageHeight": crop_h,
                    "imageWidth": crop_w,
                }
                with open(label_dir / f"{crop_name}.json", "w", encoding="utf-8") as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=2)

            total_crops += 1
            total_shapes += len(new_shapes)

    print(f"Done. crops={total_crops}, shapes_kept={total_shapes}, skipped_files={skipped_files}")
    print(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    crop_by_outline(
        roi_json_dir=r"G:\Program Files\训练平台版本更新\脚本\dianliao_mkcls\datasets\location\train\CCS\labels",
        output_dir=r"D:\Program Files\飞牛训练数据\外轮廓标注\CCS\20260606\linhy",
        target_json_dir=None,      # 外轮廓和气泡在同一份 JSON 里时留 None
        image_dir=r"G:\Program Files\训练平台版本更新\脚本\dianliao_mkcls\datasets\location\train\CCS\images",
        roi_label="outline",       # 外轮廓多边形标签名（按你的实际标签改）
        target_labels=["bubble"],  # 同步裁剪的气泡标签
        image_ext=".jpg",          # X-AnyLabeling 默认存 jpg
        min_width=0,
        min_height=0,
    )
