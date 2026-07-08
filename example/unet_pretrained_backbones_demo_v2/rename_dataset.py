"""按 00001、00002 ... 的顺序批量重命名 images 与 labels 目录。

- 不覆盖原文件：重命名后的文件会**复制**到输出目录（默认 `renamed/`）下的
  `images/`、`labels/` 子目录中，源目录保持不变。
- `images/` 中的图片按文件名自然排序后依次编号，编号宽度和起始编号可指定。
- `labels/` 目录可选：存在时与图片同名（stem 相同）的标注文件会被复制成
  相同编号，并同步更新 labelme JSON 中的 `imagePath` 字段。
- 在输出目录生成改名前后对照的 txt 文档（默认 `rename_mapping.txt`）。
"""

import argparse
import json
import re
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 images/labels 目录下的文件重命名为 00001、00002 ...，"
        "结果输出到独立目录并生成改名对照 txt。",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="包含 images（及可选 labels）文件夹的根目录",
    )
    parser.add_argument("--images-dir", type=str, default="images")
    parser.add_argument("--labels-dir", type=str, default="labels")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="renamed",
        help="输出目录，重命名后的文件与对照 txt 都放这里",
    )
    parser.add_argument(
        "--mapping-name",
        type=str,
        default="rename_mapping.txt",
        help="改名对照 txt 的文件名（位于输出目录下）",
    )
    parser.add_argument("--start", type=int, default=1, help="起始编号")
    parser.add_argument("--width", type=int, default=5, help="编号零填充位数")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划的重命名，不实际复制文件",
    )
    return parser.parse_args()


def natural_key(path: Path) -> list:
    """按自然顺序排序，使 img2 排在 img10 之前。"""
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", path.name)
    ]


def collect_images(images_dir: Path) -> list[Path]:
    images = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=natural_key)


def build_label_map(labels_dir: Path) -> dict[str, list[Path]]:
    """stem -> 该 stem 对应的所有标注文件。"""
    label_map: dict[str, list[Path]] = {}
    for path in labels_dir.iterdir():
        if path.is_file():
            label_map.setdefault(path.stem, []).append(path)
    return label_map


def copy_json_with_new_path(src: Path, dst: Path, new_image_name: str) -> None:
    """把 JSON 复制到 dst，并更新其 imagePath（保留原有目录前缀）。"""
    try:
        with src.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"[warn] 无法解析 JSON，按普通文件复制: {src.name}")
        shutil.copy2(src, dst)
        return

    if isinstance(data, dict) and "imagePath" in data:
        old_value = data.get("imagePath")
        if isinstance(old_value, str) and old_value:
            # 保留目录分隔符前缀，仅替换文件名部分
            sep_match = re.match(r"^(.*[\\/])?(.*)$", old_value)
            directory = (sep_match.group(1) or "") if sep_match else ""
            data["imagePath"] = f"{directory}{new_image_name}"
        else:
            data["imagePath"] = new_image_name

    with dst.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.width < 1:
        raise ValueError("--width 必须大于等于 1")

    root = Path(args.root)
    images_dir = root / args.images_dir
    labels_dir = root / args.labels_dir
    output_dir = Path(args.output_dir)
    out_images_dir = output_dir / args.images_dir
    out_labels_dir = output_dir / args.labels_dir

    if not images_dir.is_dir():
        raise FileNotFoundError(f"未找到 images 目录: {images_dir}")

    images = collect_images(images_dir)
    if not images:
        raise FileNotFoundError(f"images 目录中没有图片: {images_dir}")

    has_labels = labels_dir.is_dir()
    label_map = build_label_map(labels_dir) if has_labels else {}

    # (源路径, 目标路径, 是否为需要更新 imagePath 的 json, 对应新图片名)
    image_plan: list[tuple[Path, Path]] = []
    label_plan: list[tuple[Path, Path, bool, str]] = []
    # 对照记录: (类型, 原名, 新名)
    mapping_rows: list[tuple[str, str, str]] = []

    for offset, image in enumerate(images):
        number = args.start + offset
        new_stem = f"{number:0{args.width}d}"
        new_image_name = f"{new_stem}{image.suffix}"
        image_plan.append((image, out_images_dir / new_image_name))
        mapping_rows.append(("image", image.name, new_image_name))

        if has_labels:
            for label in label_map.get(image.stem, []):
                new_label_name = f"{new_stem}{label.suffix}"
                is_json = label.suffix.lower() == ".json"
                label_plan.append(
                    (label, out_labels_dir / new_label_name, is_json, new_image_name)
                )
                mapping_rows.append(("label", label.name, new_label_name))

    print(f"images 目录: {images_dir}  共 {len(image_plan)} 张图片")
    if has_labels:
        print(f"labels 目录: {labels_dir}  共 {len(label_plan)} 个标注文件")
    else:
        print("未找到 labels 目录，仅处理图片。")

    if args.dry_run:
        for _, old, new in mapping_rows:
            print(f"  {old} -> {new}")
        print("[dry-run] 未复制任何文件，也未写出对照 txt。")
        return

    out_images_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in image_plan:
        shutil.copy2(src, dst)

    if has_labels:
        out_labels_dir.mkdir(parents=True, exist_ok=True)
        for src, dst, is_json, new_image_name in label_plan:
            if is_json:
                copy_json_with_new_path(src, dst, new_image_name)
            else:
                shutil.copy2(src, dst)

    mapping_path = output_dir / args.mapping_name
    with mapping_path.open("w", encoding="utf-8") as file:
        file.write("type\told_name\tnew_name\n")
        for kind, old, new in mapping_rows:
            file.write(f"{kind}\t{old}\t{new}\n")

    print(f"输出目录: {output_dir}")
    print(f"对照文档: {mapping_path}")
    print("完成。")


if __name__ == "__main__":
    main()
