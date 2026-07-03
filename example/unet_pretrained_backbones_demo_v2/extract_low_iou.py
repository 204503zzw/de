"""根据 low_iou_report.txt 提取低 IoU 图片及同名 JSON 文件到指定目录。

用法示例::

    python extract_low_iou.py \
        --report low_iou_report.txt \
        --images-dir /path/to/images \
        --json-dir /path/to/jsons \
        --output-dir /path/to/output

    # json-dir 默认与 images-dir 相同
    python extract_low_iou.py \
        --report low_iou_report.txt \
        --images-dir /path/to/images \
        --output-dir /path/to/output
"""

import argparse
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_report(report_path: Path) -> list[tuple[str, float]]:
    """解析 low_iou_report.txt，返回 [(image_name, iou), ...]。"""
    results: list[tuple[str, float]] = []
    in_table = False
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            # 跳过空行
            if not stripped:
                continue
            # 检测表头分隔线 (---... -------)
            if stripped.startswith("---") and in_table is False:
                in_table = True
                continue
            # 表头行
            if stripped.startswith("Image") and "IoU" in stripped:
                continue
            # 解析数据行：image_name 后跟 iou 值
            if in_table:
                parts = stripped.rsplit(None, 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    try:
                        iou = float(parts[1])
                        results.append((name, iou))
                    except ValueError:
                        continue
    return results


def find_file_with_extensions(
    directory: Path, stem: str, extensions: tuple[str, ...]
) -> Path | None:
    """在目录中查找指定 stem 的文件（尝试多种扩展名）。"""
    for ext in extensions:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据 low_iou_report.txt 提取图片和同名 JSON 文件",
    )
    parser.add_argument("--report", type=str, required=True,
                        help="low_iou_report.txt 路径")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="原始图片目录")
    parser.add_argument("--json-dir", type=str, default=None,
                        help="JSON 标注文件目录（默认与 images-dir 相同）")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录，提取的文件将复制到此处")
    args = parser.parse_args()

    report_path = Path(args.report)
    images_dir = Path(args.images_dir)
    json_dir = Path(args.json_dir) if args.json_dir else images_dir
    output_dir = Path(args.output_dir)

    if not report_path.exists():
        print(f"Error: report file not found: {report_path}")
        return
    if not images_dir.exists():
        print(f"Error: images directory not found: {images_dir}")
        return

    # 创建输出子目录
    output_images = output_dir / "images"
    output_jsons = output_dir / "jsons"
    output_images.mkdir(parents=True, exist_ok=True)
    output_jsons.mkdir(parents=True, exist_ok=True)

    # 解析报告
    entries = parse_report(report_path)
    if not entries:
        print("No low IoU images found in report.")
        return

    print(f"Found {len(entries)} low IoU images in report")
    print(f"Images dir: {images_dir}")
    print(f"JSON dir:   {json_dir}")
    print(f"Output dir: {output_dir}")
    print()

    img_copied = 0
    img_missing = 0
    json_copied = 0
    json_missing = 0

    for name, iou in entries:
        # 查找图片
        img_file = find_file_with_extensions(images_dir, name, IMAGE_EXTENSIONS)
        if img_file is not None:
            shutil.copy2(img_file, output_images / img_file.name)
            img_copied += 1
        else:
            print(f"  [WARN] Image not found: {name}.*")
            img_missing += 1

        # 查找同名 JSON
        json_file = json_dir / f"{name}.json"
        if json_file.exists():
            shutil.copy2(json_file, output_jsons / json_file.name)
            json_copied += 1
        else:
            print(f"  [WARN] JSON not found: {name}.json")
            json_missing += 1

    # 汇总
    print(f"\n{'=' * 50}")
    print(f"Extraction Summary")
    print(f"{'=' * 50}")
    print(f"Total entries:  {len(entries)}")
    print(f"Images copied:  {img_copied}  (missing: {img_missing})")
    print(f"JSONs  copied:  {json_copied}  (missing: {json_missing})")
    print(f"Output:         {output_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
