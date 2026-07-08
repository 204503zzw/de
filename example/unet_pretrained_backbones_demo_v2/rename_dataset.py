"""按 00001、00002 ... 的顺序批量重命名 images 与 labels 目录。

- `images/` 中的图片按文件名排序后依次编号，编号宽度和起始编号可指定。
- `labels/` 目录可选：存在时与图片同名（stem 相同）的标注文件会被重命名成
  相同编号，并同步更新 labelme JSON 中的 `imagePath` 字段。
"""

import argparse
import json
import re
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 images/labels 目录下的文件重命名为 00001、00002 ...",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="包含 images（及可选 labels）文件夹的根目录",
    )
    parser.add_argument("--images-dir", type=str, default="images")
    parser.add_argument("--labels-dir", type=str, default="labels")
    parser.add_argument("--start", type=int, default=1, help="起始编号")
    parser.add_argument("--width", type=int, default=5, help="编号零填充位数")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划的重命名，不实际改动文件",
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


def update_image_path(json_path: Path, new_image_name: str) -> None:
    """更新 labelme JSON 中的 imagePath，保留原有目录前缀。"""
    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"[warn] 跳过非 JSON 或无法解析的标注文件: {json_path.name}")
        return

    if not isinstance(data, dict) or "imagePath" not in data:
        return

    old_value = data.get("imagePath")
    if isinstance(old_value, str) and old_value:
        # 保留目录分隔符前缀，仅替换文件名部分
        sep_match = re.match(r"^(.*[\\/])?(.*)$", old_value)
        directory = (sep_match.group(1) or "") if sep_match else ""
        data["imagePath"] = f"{directory}{new_image_name}"
    else:
        data["imagePath"] = new_image_name

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def two_phase_rename(pairs: list[tuple[Path, Path]], dry_run: bool) -> None:
    """先改成唯一临时名，再改成目标名，避免目标名与已有文件冲突。"""
    if dry_run:
        for src, dst in pairs:
            print(f"  {src.name} -> {dst.name}")
        return

    temp_pairs: list[tuple[Path, Path]] = []
    for index, (src, dst) in enumerate(pairs):
        if src == dst:
            temp_pairs.append((src, dst))
            continue
        temp = src.with_name(f".__rename_tmp_{index}__{src.name}")
        src.rename(temp)
        temp_pairs.append((temp, dst))

    for temp, dst in temp_pairs:
        if temp != dst:
            temp.rename(dst)


def main() -> None:
    args = parse_args()
    if args.width < 1:
        raise ValueError("--width 必须大于等于 1")

    root = Path(args.root)
    images_dir = root / args.images_dir
    labels_dir = root / args.labels_dir

    if not images_dir.is_dir():
        raise FileNotFoundError(f"未找到 images 目录: {images_dir}")

    images = collect_images(images_dir)
    if not images:
        raise FileNotFoundError(f"images 目录中没有图片: {images_dir}")

    has_labels = labels_dir.is_dir()
    label_map = build_label_map(labels_dir) if has_labels else {}

    image_pairs: list[tuple[Path, Path]] = []
    label_pairs: list[tuple[Path, Path]] = []
    json_updates: list[tuple[Path, str]] = []

    for offset, image in enumerate(images):
        number = args.start + offset
        new_stem = f"{number:0{args.width}d}"
        new_image_name = f"{new_stem}{image.suffix}"
        image_pairs.append((image, image.with_name(new_image_name)))

        if has_labels:
            for label in label_map.get(image.stem, []):
                new_label = label.with_name(f"{new_stem}{label.suffix}")
                label_pairs.append((label, new_label))
                if label.suffix.lower() == ".json":
                    # 记录最终 json 路径与其应写入的新图片名
                    json_updates.append((new_label, new_image_name))

    print(f"images 目录: {images_dir}  共 {len(image_pairs)} 张图片")
    two_phase_rename(image_pairs, args.dry_run)

    if has_labels:
        print(f"labels 目录: {labels_dir}  共 {len(label_pairs)} 个标注文件")
        two_phase_rename(label_pairs, args.dry_run)
        if not args.dry_run:
            for json_path, new_image_name in json_updates:
                update_image_path(json_path, new_image_name)
    else:
        print("未找到 labels 目录，仅重命名图片。")

    if args.dry_run:
        print("[dry-run] 未对文件做任何实际改动。")
    else:
        print("完成。")


if __name__ == "__main__":
    main()
