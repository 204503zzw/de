"""批量对图片文件夹中的图片及 labels 文件夹下同名 JSON 标注文件进行切片。

支持 LabelMe 格式的 JSON 标注（包含 shapes 列表，每个 shape 有 points 多边形坐标）。
切片时自动裁剪多边形到切片区域，丢弃面积过小的碎片，并更新 imageWidth / imageHeight 等字段。

用法示例::

    # 基本用法：640×640 切片，20% 重叠率
    python -m utils.slice_images_and_labels \
        --images  data/images \
        --labels  data/labels \
        --output  data/sliced \
        --slice-width  640 \
        --slice-height 640 \
        --overlap-ratio 0.2

    # 只保留含标注的切片
    python -m utils.slice_images_and_labels \
        --images data/images --labels data/labels --output data/sliced \
        --slice-width 640 --slice-height 640 --overlap-ratio 0.2 \
        --skip-empty

    # 丢弃面积占比 < 10% 的碎片标注
    python -m utils.slice_images_and_labels \
        --images data/images --labels data/labels --output data/sliced \
        --slice-width 640 --slice-height 640 --overlap-ratio 0.2 \
        --min-area-ratio 0.1
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import sys
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def polygon_area(points: list[list[float]]) -> float:
    """Shoelace formula for polygon area."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def clip_polygon_to_rect(
    points: list[list[float]],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> list[list[float]]:
    """Sutherland-Hodgman algorithm: clip polygon to axis-aligned rectangle."""

    def clip_edge(poly, edge_start, edge_end):
        """Clip polygon against one edge (half-plane)."""
        if not poly:
            return []
        ex, ey = edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]

        def inside(p):
            return ex * (p[1] - edge_start[1]) - ey * (p[0] - edge_start[0]) >= 0

        def intersect(p0, p1):
            d0x, d0y = p0[0] - edge_start[0], p0[1] - edge_start[1]
            d1x, d1y = p1[0] - edge_start[0], p1[1] - edge_start[1]
            cross0 = ex * d0y - ey * d0x
            cross1 = ex * d1y - ey * d1x
            denom = cross0 - cross1
            if abs(denom) < 1e-12:
                return p0
            t = cross0 / denom
            return [p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])]

        result = []
        for i in range(len(poly)):
            cur = poly[i]
            nxt = poly[(i + 1) % len(poly)]
            cur_in = inside(cur)
            nxt_in = inside(nxt)
            if cur_in:
                result.append(cur)
                if not nxt_in:
                    result.append(intersect(cur, nxt))
            elif nxt_in:
                result.append(intersect(cur, nxt))
        return result

    edges = [
        ([x_min, y_min], [x_max, y_min]),  # top
        ([x_max, y_min], [x_max, y_max]),  # right
        ([x_max, y_max], [x_min, y_max]),  # bottom
        ([x_min, y_max], [x_min, y_min]),  # left
    ]
    poly = [list(p) for p in points]
    for e_start, e_end in edges:
        poly = clip_edge(poly, e_start, e_end)
        if not poly:
            return []
    return poly


def clip_rectangle_to_rect(
    points: list[list[float]],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> list[list[float]]:
    """Clip a rectangle (2-point shape) to the tile bounds."""
    if len(points) != 2:
        return points
    rx0 = max(min(points[0][0], points[1][0]), x_min)
    ry0 = max(min(points[0][1], points[1][1]), y_min)
    rx1 = min(max(points[0][0], points[1][0]), x_max)
    ry1 = min(max(points[0][1], points[1][1]), y_max)
    if rx1 <= rx0 or ry1 <= ry0:
        return []
    return [[rx0, ry0], [rx1, ry1]]


def generate_tile_coords(
    img_width: int,
    img_height: int,
    tile_w: int,
    tile_h: int,
    overlap_ratio: float = 0.0,
) -> list[tuple[int, int, int, int]]:
    """生成切片坐标列表 [(x0, y0, x1, y1), ...]，覆盖整张图片。

    Args:
        overlap_ratio: 相邻切片重叠率，范围 [0, 1)，例如 0.2 表示 20% 重叠。

    边界策略：
    - 当原图尺寸 >= 切片尺寸时，最后一列/行的切片向前回退，保证每个切片都是
      完整的 tile_w × tile_h（回退部分与前一个切片重叠）。
    - 当原图某一边 < 切片尺寸时，不回退、不填充，直接保留原图该边的实际尺寸，
      生成一个宽/高小于 tile 的切片。
    """
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError(
            f"overlap_ratio ({overlap_ratio}) must be in [0, 1)"
        )
    stride_x = max(1, int(tile_w * (1.0 - overlap_ratio)))
    stride_y = max(1, int(tile_h * (1.0 - overlap_ratio)))

    tiles = []

    # 原图 >= 切片尺寸：正常滑窗；原图 < 切片尺寸：仅从 0 开始，保留原尺寸
    y_starts = list(range(0, max(img_height - tile_h, 0) + 1, stride_y))
    x_starts = list(range(0, max(img_width - tile_w, 0) + 1, stride_x))

    # 确保最后一个切片覆盖到图片边界（回退到 img_dim - tile_dim 的位置）
    # 当 img_dim < tile_dim 时 max(..., 0) == 0，不会回退，保留原尺寸
    # 当回退位置与上一个位置间距 < stride 时，替换上一个而非新增，避免近重复切片
    y_boundary = max(img_height - tile_h, 0)
    if not y_starts or y_starts[-1] + tile_h < img_height:
        if len(y_starts) >= 2 and y_boundary - y_starts[-1] < stride_y:
            y_starts[-1] = y_boundary
        else:
            y_starts.append(y_boundary)

    x_boundary = max(img_width - tile_w, 0)
    if not x_starts or x_starts[-1] + tile_w < img_width:
        if len(x_starts) >= 2 and x_boundary - x_starts[-1] < stride_x:
            x_starts[-1] = x_boundary
        else:
            x_starts.append(x_boundary)

    # 去重
    y_starts = sorted(set(y_starts))
    x_starts = sorted(set(x_starts))

    for row, y0 in enumerate(y_starts):
        for col, x0 in enumerate(x_starts):
            x1 = min(x0 + tile_w, img_width)
            y1 = min(y0 + tile_h, img_height)
            tiles.append((x0, y0, x1, y1, row, col))
    return tiles


def slice_label(
    label_data: dict,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    min_area_ratio: float = 0.0,
) -> dict | None:
    """将 LabelMe JSON 标注裁剪到 (x0, y0, x1, y1) 区域。

    返回新的 label dict（坐标已平移到切片坐标系），如果没有有效 shape 则返回 None。
    """
    tile_w = x1 - x0
    tile_h = y1 - y0

    new_shapes = []
    for shape in label_data.get("shapes", []):
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "polygon")

        if shape_type == "rectangle" and len(points) == 2:
            clipped = clip_rectangle_to_rect(points, x0, y0, x1, y1)
            if not clipped:
                continue
            # 计算面积比
            orig_w = abs(points[1][0] - points[0][0])
            orig_h = abs(points[1][1] - points[0][1])
            orig_area = orig_w * orig_h
            new_w = abs(clipped[1][0] - clipped[0][0])
            new_h = abs(clipped[1][1] - clipped[0][1])
            new_area = new_w * new_h
            if orig_area > 0 and new_area / orig_area < min_area_ratio:
                continue
            shifted = [[p[0] - x0, p[1] - y0] for p in clipped]
        elif shape_type in ("polygon", "linestrip", "line"):
            if len(points) < 2:
                continue
            if shape_type in ("polygon",) and len(points) >= 3:
                orig_area = polygon_area(points)
                clipped = clip_polygon_to_rect(points, x0, y0, x1, y1)
                if len(clipped) < 3:
                    continue
                new_area = polygon_area(clipped)
                # 丢弃退化为线/点的裁剪结果（面积 ≈ 0）
                if new_area < 1.0:
                    continue
                if orig_area > 0 and new_area / orig_area < min_area_ratio:
                    continue
            else:
                # 线段类型：只保留在切片区域内的点
                clipped = [
                    p for p in points
                    if x0 <= p[0] <= x1 and y0 <= p[1] <= y1
                ]
                if len(clipped) < 2:
                    continue
            shifted = [[p[0] - x0, p[1] - y0] for p in clipped]
        elif shape_type == "point":
            if len(points) != 1:
                continue
            px, py = points[0]
            if not (x0 <= px <= x1 and y0 <= py <= y1):
                continue
            shifted = [[px - x0, py - y0]]
        elif shape_type == "circle":
            if len(points) != 2:
                continue
            cx, cy = points[0]
            ex, ey = points[1]
            r = ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5
            # 粗略检查：圆的外接矩形是否与切片有交集
            if cx + r < x0 or cx - r > x1 or cy + r < y0 or cy - r > y1:
                continue
            shifted = [[cx - x0, cy - y0], [ex - x0, ey - y0]]
        else:
            # 未知类型：检查所有点是否在切片区域内
            inside = [
                p for p in points
                if x0 <= p[0] <= x1 and y0 <= p[1] <= y1
            ]
            if not inside:
                continue
            shifted = [[p[0] - x0, p[1] - y0] for p in inside]

        new_shape = copy.deepcopy(shape)
        new_shape["points"] = shifted
        new_shapes.append(new_shape)

    if not new_shapes:
        return None

    new_label = copy.deepcopy(label_data)
    new_label["shapes"] = new_shapes
    new_label["imageWidth"] = tile_w
    new_label["imageHeight"] = tile_h
    # 置空原图 imageData（保留字段，X-AnyLabeling 等工具需要该字段存在）
    new_label["imageData"] = None
    return new_label


def encode_image_data(image: Image.Image, fmt: str = "PNG") -> str:
    """将 PIL Image 编码为 base64 字符串（LabelMe imageData 格式）。"""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def process_one_image(
    image_path: Path,
    label_path: Path | None,
    output_images_dir: Path,
    output_labels_dir: Path,
    tile_w: int,
    tile_h: int,
    overlap_ratio: float,
    min_area_ratio: float,
    skip_empty: bool,
    embed_image_data: bool,
) -> tuple[int, int]:
    """处理单张图片及其标注，返回 (生成切片数, 跳过切片数)。"""
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size

    label_data = None
    if label_path is not None and label_path.is_file():
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

    tiles = generate_tile_coords(img_width, img_height, tile_w, tile_h, overlap_ratio)
    stem = image_path.stem
    suffix = image_path.suffix

    created = 0
    skipped = 0

    for x0, y0, x1, y1, row, col in tiles:
        tile_name = f"{stem}_r{row}_c{col}"

        new_label = None
        if label_data is not None:
            new_label = slice_label(label_data, x0, y0, x1, y1, min_area_ratio)

        if skip_empty and new_label is None and label_data is not None:
            skipped += 1
            continue

        # 保存切片图片（统一输出 PNG 格式）
        tile_img = img.crop((x0, y0, x1, y1))
        tile_img_path = output_images_dir / f"{tile_name}.png"
        tile_img.save(tile_img_path, format="PNG")

        # 保存切片标注
        if label_data is not None:
            if new_label is None:
                # 空标注也写出（skip_empty=False 时）
                new_label = copy.deepcopy(label_data)
                new_label["shapes"] = []
                new_label["imageWidth"] = x1 - x0
                new_label["imageHeight"] = y1 - y0
                new_label["imageData"] = None

            new_label["imagePath"] = f"{tile_name}.png"
            if embed_image_data:
                new_label["imageData"] = encode_image_data(tile_img)

            tile_label_path = output_labels_dir / f"{tile_name}.json"
            with open(tile_label_path, "w", encoding="utf-8") as f:
                json.dump(new_label, f, ensure_ascii=False, indent=2)

        created += 1

    return created, skipped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="批量切片图片及 LabelMe JSON 标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--images", required=True, type=Path,
        help="输入图片文件夹路径",
    )
    parser.add_argument(
        "--labels", required=True, type=Path,
        help="输入 LabelMe JSON 标注文件夹路径",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="输出目录（自动创建 images/ 和 labels/ 子文件夹）",
    )
    parser.add_argument(
        "--slice-width", required=True, type=int,
        help="切片宽度（像素）",
    )
    parser.add_argument(
        "--slice-height", required=True, type=int,
        help="切片高度（像素）",
    )
    parser.add_argument(
        "--overlap-ratio", type=float, default=0.0,
        help="相邻切片重叠率，范围 [0, 1)，例如 0.2 表示 20%% 重叠（默认 0，不重叠）",
    )
    parser.add_argument(
        "--min-area-ratio", type=float, default=0.0,
        help="标注面积裁剪后与原始面积之比的最小阈值，低于此值丢弃该标注（默认 0，不丢弃）",
    )
    parser.add_argument(
        "--skip-empty", action="store_true",
        help="跳过不含任何标注的切片（不生成图片和 JSON）",
    )
    parser.add_argument(
        "--embed-image-data", action="store_true",
        help="在输出 JSON 中内嵌 base64 图片数据（imageData 字段）",
    )
    args = parser.parse_args(argv)

    images_dir: Path = args.images
    labels_dir: Path = args.labels
    output_dir: Path = args.output

    if not images_dir.is_dir():
        print(f"错误：图片文件夹不存在: {images_dir}", file=sys.stderr)
        sys.exit(1)
    if not labels_dir.is_dir():
        print(f"错误：标注文件夹不存在: {labels_dir}", file=sys.stderr)
        sys.exit(1)

    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"警告：图片文件夹中未找到图片: {images_dir}", file=sys.stderr)
        sys.exit(0)

    total_created = 0
    total_skipped = 0
    total_no_label = 0

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.json"
        if not label_path.is_file():
            total_no_label += 1
            label_path = None

        created, skipped = process_one_image(
            image_path=img_path,
            label_path=label_path,
            output_images_dir=output_images_dir,
            output_labels_dir=output_labels_dir,
            tile_w=args.slice_width,
            tile_h=args.slice_height,
            overlap_ratio=args.overlap_ratio,
            min_area_ratio=args.min_area_ratio,
            skip_empty=args.skip_empty,
            embed_image_data=args.embed_image_data,
        )
        total_created += created
        total_skipped += skipped
        print(
            f"  {img_path.name}: 生成 {created} 个切片"
            + (f"，跳过 {skipped} 个空切片" if skipped else "")
        )

    print(f"\n完成！共处理 {len(image_files)} 张图片，"
          f"生成 {total_created} 个切片，跳过 {total_skipped} 个空切片。")
    if total_no_label:
        print(f"  （其中 {total_no_label} 张图片无对应 JSON 标注）")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
