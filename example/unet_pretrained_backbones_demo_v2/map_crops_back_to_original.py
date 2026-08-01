"""Map labels annotated on cropped ROI images back to the original image coordinates.

Workflow
--------
1. Original images have ROI (rectangle / polygon / rotation) annotations in LabelMe JSON.
2. Each ROI was cropped into a sub-image named ``{original_stem}_{index}.png`` or
   ``{original_stem}_{index}-{shape_type}.jpg``（X-AnyLabeling 命名），``index`` 从 1 开始，
   与原图 JSON 中 ROI 形状顺序一致。
3. 这些裁剪图被标注了 outline / bubble 等多边形。
4. 本脚本把这些标注**按裁剪时所用变换的逆变换**映射回原图坐标系。

反仿射（inverse affine）
-----------------------
裁剪有两种方式，映射回去就有两种逆变换，均写成 2x3 仿射矩阵 ``M: 原图 -> 裁剪图``，
回映射用 ``M^-1``：

- ``bbox``（正外接矩形直接切片，无旋转）::

      M = [[sx, 0, -sx*off_x],
           [0, sy, -sy*off_y]]      # sx = crop_w / src_w, sy = crop_h / src_h

- ``affine``（默认；ROI 做了仿射拉正 warpAffine），与仿射裁剪脚本 ``affine_crop`` 逐行对齐::

      tl,tr,br,bl = order_points(四角)    # tl=min(x+y), tr=min(y-x), br=max(x+y), bl=max(y-x)
      w = round(|tr-tl|), h = round(|bl-tl|)
      M = getAffineTransform([tl, tr, br] -> [(0,0), (w-1,0), (w-1,h-1)])
      再乘上标注尺寸/拉正尺寸的缩放

  两个容易踩的点：角点排序必须用上面的 min/max 规则（绕质心按角度排序在接近 45° 时会
  整体差 90°，表现为“逆拉正角度不对”）；目标角点是 ``w-1`` / ``h-1`` 而不是 ``w`` / ``h``。

``roi_mode="auto"`` 时按 ROI 形状自动选择：``shape_type == "rotation"``、或 ROI 明显是斜的
（最小外接矩形面积 / 正外接矩形面积 < ``rotated_ratio``）-> ``affine``，否则 ``bbox``。
ROI 是任意点数的多边形时，拉正四角取其**最小外接矩形**（纯 Python 的旋转卡壳实现，
等价于 ``cv2.minAreaRect``）。

``roi_label`` 必须与裁剪脚本的 ``roi_label`` 一致：裁剪时只对该标签的形状编号 ``_1 _2 ...``，
这里若把 JSON 里所有形状都算进去，索引就会错位到别的 ROI 上。

纯标准库实现（PIL 仅在需要读裁剪图尺寸时可选使用），不依赖 OpenCV。
"""

import argparse
import json
import math
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
AXIS_ALIGNED_TOL = 1e-3


# --------------------------------------------------------------------------------------
# 仿射工具（2x3 矩阵，行优先: (a, b, c, d, e, f) 表示 x' = a*x + b*y + c, y' = d*x + e*y + f）
# --------------------------------------------------------------------------------------

def affine_from_points(src_pts, dst_pts):
    """由 3 对点求仿射矩阵 src -> dst，退化时返回 None。"""
    (x1, y1), (x2, y2), (x3, y3) = [(float(p[0]), float(p[1])) for p in src_pts]
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if abs(det) < 1e-9:
        return None

    def solve(u1, u2, u3):
        a = ((u2 - u1) * (y3 - y1) - (u3 - u1) * (y2 - y1)) / det
        b = ((u3 - u1) * (x2 - x1) - (u2 - u1) * (x3 - x1)) / det
        c = u1 - a * x1 - b * y1
        return a, b, c

    a, b, c = solve(*[float(p[0]) for p in dst_pts])
    d, e, f = solve(*[float(p[1]) for p in dst_pts])
    return (a, b, c, d, e, f)


def invert_affine(m):
    """求 2x3 仿射矩阵的逆，退化时返回 None。"""
    a, b, c, d, e, f = m
    det = a * e - b * d
    if abs(det) < 1e-12:
        return None
    ia, ib = e / det, -b / det
    id_, ie = -d / det, a / det
    ic = -(ia * c + ib * f)
    if_ = -(id_ * c + ie * f)
    return (ia, ib, ic, id_, ie, if_)


def apply_affine(m, x, y):
    a, b, c, d, e, f = m
    return a * x + b * y + c, d * x + e * y + f


def scale_affine(m, sx, sy):
    """在 m 之后再做一次缩放: (sx, sy) ∘ m。"""
    a, b, c, d, e, f = m
    return (a * sx, b * sx, c * sx, d * sy, e * sy, f * sy)


# --------------------------------------------------------------------------------------
# ROI 解析
# --------------------------------------------------------------------------------------

def parse_cropped_filename(filename_stem):
    """解析 ``{original_stem}_{roi_index}`` 或 ``{original_stem}_{roi_index}-{shape_type}``。

    返回 ``(original_stem, roi_index)``（1-based），不匹配时返回 ``(None, None)``。
    """
    stem = filename_stem
    dash = stem.rfind("-")
    if dash > 0 and stem[dash + 1:].isalpha():
        stem = stem[:dash]
    idx = stem.rfind("_")
    if idx <= 0:
        return None, None
    suffix = stem[idx + 1:]
    if not suffix.isdigit():
        return None, None
    return stem[:idx], int(suffix)


def get_roi_bbox(shape):
    """Extract the bounding box ``(x_min, y_min, x_max, y_max)`` from a LabelMe shape."""
    points = shape.get("points", [])
    if len(points) < 2:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def convex_hull(points):
    """Andrew monotone chain，返回逆时针凸包（不含重复端点）。"""
    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def min_area_rect_quad(points):
    """最小外接矩形的四个角（旋转卡壳：枚举凸包每条边作为矩形一边）。

    与 ``cv2.minAreaRect`` + ``cv2.boxPoints`` 等价，纯标准库实现。
    """
    hull = convex_hull(points)
    if len(hull) < 3:
        return None

    best = None
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        edge = math.hypot(x2 - x1, y2 - y1)
        if edge < 1e-9:
            continue
        ux, uy = (x2 - x1) / edge, (y2 - y1) / edge
        vx, vy = -uy, ux
        us = [p[0] * ux + p[1] * uy for p in hull]
        vs = [p[0] * vx + p[1] * vy for p in hull]
        u_min, u_max = min(us), max(us)
        v_min, v_max = min(vs), max(vs)
        area = (u_max - u_min) * (v_max - v_min)
        if best is None or area < best[0] - 1e-9:
            best = (area, (ux, uy), (vx, vy), u_min, u_max, v_min, v_max)

    if best is None:
        return None
    _, (ux, uy), (vx, vy), u_min, u_max, v_min, v_max = best

    def corner(u, v):
        return (u * ux + v * vx, u * uy + v * vy)

    return order_quad(
        [
            corner(u_min, v_min),
            corner(u_max, v_min),
            corner(u_max, v_max),
            corner(u_min, v_max),
        ]
    )


def polygon_area(points):
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) < 3:
        return 0.0
    total = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def order_quad(points):
    """把四点排成 tl, tr, br, bl，与裁剪脚本的 ``order_points`` 完全一致。

    tl = min(x+y), tr = min(y-x), br = max(x+y), bl = max(y-x)。
    注意不能改成“绕质心按角度排序 + 选起点”，那样在接近 45° 时会整体差 90°，
    逆变换的旋转角就对不上。
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    tl = min(pts, key=lambda p: p[0] + p[1])
    tr = min(pts, key=lambda p: p[1] - p[0])
    br = max(pts, key=lambda p: p[0] + p[1])
    bl = max(pts, key=lambda p: p[1] - p[0])
    return [tl, tr, br, bl]


def is_axis_aligned(quad):
    xs = sorted({round(p[0], 3) for p in quad})
    ys = sorted({round(p[1], 3) for p in quad})
    return len(xs) <= 2 and len(ys) <= 2


def rectified_size(quad):
    """拉正后的宽高，与裁剪脚本 ``affine_crop`` 一致：四舍五入且至少 1。"""
    tl, tr, _, bl = quad
    width = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
    height = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
    return max(int(round(width)), 1), max(int(round(height)), 1)


def bbox_forward_affine(bbox, bbox_mode="floor_ceil"):
    """正外接矩形裁剪的正向仿射（原图 -> 裁剪图）及裁剪源区域尺寸。"""
    x_min, y_min, x_max, y_max = bbox
    if bbox_mode == "anylabeling":
        # 与 X-AnyLabeling 一致: cv2.boundingRect(np.int32(points))
        off_x, off_y = int(x_min), int(y_min)
        src_w = int(x_max) - off_x + 1
        src_h = int(y_max) - off_y + 1
    else:
        off_x, off_y = math.floor(x_min), math.floor(y_min)
        src_w = math.ceil(x_max) - off_x
        src_h = math.ceil(y_max) - off_y
    return (1.0, 0.0, -float(off_x), 0.0, 1.0, -float(off_y)), src_w, src_h


def affine_forward_affine(quad):
    """仿射拉正裁剪的正向仿射（原图 -> 拉正裁剪图）及拉正尺寸。

    与裁剪脚本逐行对应::

        dst = [(0, 0), (w - 1, 0), (w - 1, h - 1)]
        M   = cv2.getAffineTransform(src[:3] = [tl, tr, br], dst)

    注意目标角点用的是 ``w-1`` / ``h-1``（不是 w / h），少了这一格会整体差一个
    ``w/(w-1)`` 的尺度。
    """
    tl, tr, br, _ = quad
    w, h = rectified_size(quad)
    if w <= 1 or h <= 1:
        return None, 0, 0
    m = affine_from_points(
        [tl, tr, br],
        [(0.0, 0.0), (float(w - 1), 0.0), (float(w - 1), float(h - 1))],
    )
    return m, w, h


def rect2_to_quad(p1, p2):
    """两点矩形 -> 四角（与裁剪脚本 ``rect2_to_4pts`` 一致）。"""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def roi_quad(points, shape_type="polygon"):
    """ROI 的拉正四角，与裁剪脚本 ``roi_to_4pts`` + ``order_points`` 一致。"""
    if shape_type == "rectangle" and len(points) == 2:
        return order_quad(rect2_to_quad(points[0], points[1]))
    if len(points) == 4:
        return order_quad(points)
    if len(points) >= 3:
        return min_area_rect_quad(points)
    return None


def build_forward_transform(roi_shape, roi_mode="affine", bbox_mode="floor_ceil", rotated_ratio=0.98):
    """返回 ``(M, src_w, src_h)``，M 为原图 -> 裁剪图（未含标注端 resize）的仿射。

    ``roi_mode="auto"``：rotation 形状、或最小外接矩形明显小于正外接矩形
    （面积比 < *rotated_ratio*，即 ROI 是斜的）时走反仿射拉正，否则走正外接矩形。
    """
    points = roi_shape.get("points", [])
    shape_type = roi_shape.get("shape_type", "polygon")
    quad = roi_quad(points, shape_type)

    use_affine = False
    if roi_mode == "affine":
        use_affine = True
    elif roi_mode == "auto" and quad is not None:
        if shape_type == "rotation":
            use_affine = True
        elif not is_axis_aligned(quad):
            bbox = get_roi_bbox(roi_shape)
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if bbox else 0.0
            quad_area = polygon_area(quad)
            if bbox_area > 0 and quad_area / bbox_area < rotated_ratio:
                use_affine = True

    if use_affine:
        if quad is None:
            return None, 0, 0
        return affine_forward_affine(quad)

    bbox = get_roi_bbox(roi_shape)
    if bbox is None:
        return None, 0, 0
    return bbox_forward_affine(bbox, bbox_mode=bbox_mode)


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------

def load_roi_data(roi_json_dir, roi_label=None):
    """Load all ROI JSONs, indexed by file stem -> ``(json_data, [roi_shape, ...])``.

    *roi_label* 非空时只保留该标签的形状 —— 必须与裁剪脚本的 ``roi_label``
    一致，否则文件名里的 ``_{idx}`` 对不上同一个 ROI。
    """
    roi_json_dir = Path(roi_json_dir)
    roi_map = {}
    for json_path in sorted(roi_json_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        shapes = list(data.get("shapes", []))
        if roi_label:
            shapes = [s for s in shapes if s.get("label", "") == roi_label]
        roi_map[json_path.stem] = (data, shapes)
    return roi_map


def get_image_size(image_path):
    """Return ``(width, height)`` of an image, or ``None`` if it cannot be read."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(image_path) as img:
            return img.size
    except (OSError, ValueError):
        return None


def find_image_file(directory, stem):
    """Find an image file named ``{stem}.<ext>`` inside *directory*."""
    directory = Path(directory)
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def resolve_crop_size(outline_data, crop_stem, crop_dir, src_w, src_h):
    """标注时裁剪图的 ``(width, height)``。

    优先级: outline JSON 的 imageWidth/imageHeight -> 裁剪图文件 -> 裁剪源区域尺寸（视为未缩放）。
    """
    crop_w = float(outline_data.get("imageWidth") or 0)
    crop_h = float(outline_data.get("imageHeight") or 0)

    if (crop_w <= 0 or crop_h <= 0) and crop_dir is not None:
        crop_image = find_image_file(crop_dir, crop_stem)
        if crop_image is not None:
            size = get_image_size(crop_image)
            if size is not None:
                crop_w, crop_h = float(size[0]), float(size[1])

    if crop_w <= 0 or crop_h <= 0:
        crop_w, crop_h = float(src_w), float(src_h)

    return crop_w, crop_h


def map_shape(shape, inv_matrix):
    """Return a copy of *shape* with points mapped back to original coordinates."""
    new_shape = dict(shape)
    mapped = []
    for p in shape.get("points", []):
        x, y = apply_affine(inv_matrix, float(p[0]), float(p[1]))
        mapped.append([round(x, 2), round(y, 2)])
    new_shape["points"] = mapped
    return new_shape


def find_original_image(roi_json_data, image_search_dir):
    """Locate the original image referenced by a ROI JSON. Returns Path or None."""
    stem = Path(roi_json_data.get("imagePath", "")).stem or None
    if stem is None:
        return None
    return find_image_file(image_search_dir, stem)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def run(
    roi_json_dir,
    outline_json_dir,
    output_dir,
    keep_roi=False,
    image_dir=None,
    crop_dir=None,
    roi_label=None,
    roi_mode="affine",
    bbox_mode="floor_ceil",
    rotated_ratio=0.98,
):
    roi_json_dir = Path(roi_json_dir)
    outline_json_dir = Path(outline_json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_search_dir = Path(image_dir) if image_dir else roi_json_dir
    crop_dir = Path(crop_dir) if crop_dir else None

    roi_map = load_roi_data(roi_json_dir, roi_label=roi_label)
    if not roi_map:
        print(f"No ROI JSON files found in {roi_json_dir}")
        return

    result_shapes = {}
    matched = 0
    skipped = 0

    for outline_json_path in sorted(outline_json_dir.glob("*.json")):
        crop_stem = outline_json_path.stem
        original_stem, roi_index = parse_cropped_filename(crop_stem)
        if original_stem is None or roi_index is None:
            print(f"Skip {outline_json_path.name}: filename does not match '{{stem}}_{{index}}' pattern")
            skipped += 1
            continue

        if original_stem not in roi_map:
            print(f"Skip {outline_json_path.name}: no ROI JSON found for '{original_stem}'")
            skipped += 1
            continue

        roi_data, roi_shapes = roi_map[original_stem]

        if roi_index < 1 or roi_index > len(roi_shapes):
            print(
                f"Skip {outline_json_path.name}: ROI index {roi_index} out of range "
                f"(original has {len(roi_shapes)} ROI shapes)"
            )
            skipped += 1
            continue

        roi_shape = roi_shapes[roi_index - 1]  # 1-based -> 0-based
        forward, src_w, src_h = build_forward_transform(
            roi_shape, roi_mode=roi_mode, bbox_mode=bbox_mode, rotated_ratio=rotated_ratio
        )
        if forward is None or src_w <= 0 or src_h <= 0:
            print(f"Skip {outline_json_path.name}: cannot build crop transform from ROI shape")
            skipped += 1
            continue

        with open(outline_json_path, "r", encoding="utf-8") as f:
            outline_data = json.load(f)

        crop_w, crop_h = resolve_crop_size(outline_data, crop_stem, crop_dir, src_w, src_h)
        # 标注端若做过 resize，把这一步缩放并入正向仿射，再整体求逆
        forward = scale_affine(forward, crop_w / src_w, crop_h / src_h)
        inverse = invert_affine(forward)
        if inverse is None:
            print(f"Skip {outline_json_path.name}: degenerate crop transform")
            skipped += 1
            continue

        outline_shapes = outline_data.get("shapes", [])
        if not outline_shapes:
            print(f"Note {outline_json_path.name}: no outline shapes found")

        for shape in outline_shapes:
            result_shapes.setdefault(original_stem, []).append(map_shape(shape, inverse))

        matched += 1

    written = 0
    copied_images = 0
    for stem, (roi_data, roi_shapes) in roi_map.items():
        mapped_outlines = result_shapes.get(stem, [])
        if not mapped_outlines and not keep_roi:
            continue

        output_shapes = []
        if keep_roi:
            output_shapes.extend(roi_shapes)
        output_shapes.extend(mapped_outlines)

        output_data = {
            "version": roi_data.get("version", "5.0.1"),
            "flags": roi_data.get("flags", {}),
            "shapes": output_shapes,
            "imagePath": roi_data.get("imagePath", ""),
            "imageData": None,
            "imageHeight": roi_data.get("imageHeight"),
            "imageWidth": roi_data.get("imageWidth"),
        }

        output_path = output_dir / f"{stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        written += 1

        src_image = find_original_image(roi_data, image_search_dir)
        if src_image is not None:
            dst_image = output_dir / src_image.name
            if dst_image.resolve() != src_image.resolve():
                shutil.copy2(src_image, dst_image)
                copied_images += 1
        else:
            print(f"Warning: image not found for '{stem}', skipping image copy")

    print(f"Done. matched_outlines={matched}, skipped={skipped}, written_jsons={written}, copied_images={copied_images}")
    print(f"Output: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Map labels from cropped ROI images back to original image coordinates (inverse affine).",
    )
    parser.add_argument("--roi-json-dir", required=True, help="Directory of original image LabelMe JSONs with ROI annotations.")
    parser.add_argument("--outline-json-dir", required=True, help="Directory of cropped ROI image LabelMe JSONs with outline annotations.")
    parser.add_argument("--output-dir", required=True, help="Output directory for mapped LabelMe JSONs.")
    parser.add_argument("--image-dir", default=None, help="Directory of original images. Defaults to --roi-json-dir.")
    parser.add_argument("--crop-dir", default=None, help="Directory of cropped images, used to read crop size when the outline JSON lacks imageWidth/imageHeight.")
    parser.add_argument("--keep-roi", action="store_true", help="Keep the original ROI shapes in the output JSON.")
    parser.add_argument("--roi-label", default=None, help="ROI 标签名，需与裁剪脚本的 --roi-label 一致（不填则用全部形状）。")
    parser.add_argument("--roi-mode", default="affine", choices=["auto", "bbox", "affine"], help="Crop transform used: bbox slice, affine rectification, or auto-detect.")
    parser.add_argument("--rotated-ratio", type=float, default=0.98, help="auto 模式判定 ROI 为斜框的面积比阈值（minAreaRect / bbox）。")
    parser.add_argument("--bbox-mode", default="floor_ceil", choices=["floor_ceil", "anylabeling"], help="bbox 裁剪取整方式：floor/ceil 或 X-AnyLabeling 的 int32 boundingRect。")
    return parser.parse_args()


if __name__ == "__main__":
    run(
        roi_json_dir=r"G:\Program Files\训练平台版本更新\数据集\hangda\outline_labels",
        outline_json_dir=r"D:\Program Files\已标注数据集\飞牛\20260718_crop\merge\labels_",
        output_dir=r"G:\Program Files\训练平台版本更新\数据集\hangda\bubble_labels_",
        image_dir=r"G:\Program Files\训练平台版本更新\数据集\hangda\images",
        keep_roi=False,
        roi_label="ccs_roi",     # 必须与裁剪脚本的 roi_label 一致
        roi_mode="affine",          # 斜的多边形/rotation -> 反仿射拉正（多边形取 minAreaRect）；正框 -> 正外接矩形
        bbox_mode="floor_ceil",   # 裁剪脚本用 X-AnyLabeling 方式时改成 "anylabeling"
    )
