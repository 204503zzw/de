import argparse
import json
import random
from pathlib import Path

from common import (
    IMAGE_EXCLUDE_DIRS,
    IMAGE_EXTENSIONS,
    MASK_EXCLUDE_DIRS,
    MASK_EXTENSIONS,
    MASK_PREFER_DIRS,
    build_file_maps,
    ensure_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--masks-dir", type=str, required=False, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-filename", action="store_true")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归分层模式：--images-dir 视为数据集根，递归查找每个 <images-subdir> 文件夹，"
        "在同级 <masks-subdir> 文件夹里按文件名 stem 配对 mask。token 用相对路径(避免跨目录同名冲突)，"
        "并额外写出 mask_index.json 记录精确的 图片->mask 配对。",
    )
    parser.add_argument("--images-subdir", type=str, default="images", help="递归模式下图片子目录名(默认 images)。")
    parser.add_argument(
        "--train-list",
        type=str,
        default=None,
        help="指定划分：train 样本名单(每行一个 stem/文件名/相对路径)，代替按 ratio 随机划分。",
    )
    parser.add_argument(
        "--val-list",
        type=str,
        default=None,
        help="指定划分：val 样本名单，需与 --train-list 同时使用。",
    )
    parser.add_argument("--masks-subdir", type=str, default="labels", help="递归模式下 mask 子目录名(默认 labels)。")
    return parser.parse_args()


def collect_flat_stems(images_dir: str, masks_dir: str):
    """原始平铺模式：images-dir / masks-dir 各自递归扫描，按 stem 求交集配对。"""
    image_by_name, image_by_stem = build_file_maps(
        images_dir,
        exclude_parent_dirs=IMAGE_EXCLUDE_DIRS,
    )
    mask_by_name, mask_by_stem = build_file_maps(
        masks_dir,
        MASK_EXTENSIONS,
        prefer_parent_dirs=MASK_PREFER_DIRS,
        exclude_parent_dirs=MASK_EXCLUDE_DIRS,
    )
    common_stems = sorted(set(image_by_stem) & set(mask_by_stem))
    if not common_stems:
        raise FileNotFoundError("No matching image/mask pairs found.")
    return image_by_stem, common_stems


def collect_hierarchical_pairs(root: Path, images_subdir: str, masks_subdir: str):
    """递归分层模式：找到每个 <images_subdir> 文件夹，在同级 <masks_subdir> 里按 stem 配对。

    返回 (pairs, unmatched)：
      - pairs: list of (rel_image_token, abs_image_path, abs_mask_path)
      - unmatched: 找不到同名 mask 的图片路径列表(用于告警)
    """
    pairs = []
    unmatched = []
    for images_folder in sorted(p for p in root.rglob("*") if p.is_dir() and p.name == images_subdir):
        masks_folder = images_folder.parent / masks_subdir
        mask_by_stem = {}
        if masks_folder.is_dir():
            # 同一 stem 若既有图片又有 LabelMe json，优先图片(已渲染的 mask)。
            candidates = [
                p
                for p in masks_folder.iterdir()
                if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS
            ]
            for mask_path in sorted(candidates, key=lambda p: (p.suffix.lower() == ".json", p.name)):
                mask_by_stem.setdefault(mask_path.stem, mask_path)
        for image_path in sorted(images_folder.iterdir()):
            if not (image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS):
                continue
            mask_path = mask_by_stem.get(image_path.stem)
            if mask_path is None:
                unmatched.append(image_path)
                continue
            token = image_path.relative_to(root).as_posix()
            pairs.append((token, image_path.resolve(), mask_path.resolve()))
    return pairs, unmatched


def read_name_list(path: str) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig") as file:
        return [line.strip() for line in file if line.strip()]


def split_by_lists(
    tokens: list[str],
    train_list: str,
    val_list: str,
) -> tuple[set[str], set[str]]:
    """按给定名单划分。名单里可写 stem / 文件名 / 相对路径，统一按 stem 匹配 token。"""
    by_stem: dict[str, list[str]] = {}
    for token in tokens:
        by_stem.setdefault(Path(token).stem, []).append(token)

    selected: dict[str, set[str]] = {}
    for name, list_path in (("train", train_list), ("val", val_list)):
        keep: set[str] = set()
        missing: list[str] = []
        for entry in read_name_list(list_path):
            matches = by_stem.get(Path(entry).stem)
            if not matches:
                missing.append(entry)
                continue
            if len(matches) > 1:
                raise ValueError(
                    f"名单项 '{entry}' 对应多个样本，无法唯一匹配: {matches}。"
                    "请在名单里改写相对路径，或先消除重名。"
                )
            keep.add(matches[0])
        if missing:
            print(f"Warning: {name} 名单中 {len(missing)} 个名字在数据集里找不到配对样本，已忽略，例如:")
            for entry in missing[:10]:
                print(f"  {entry}")
        selected[name] = keep

    overlap = selected["train"] & selected["val"]
    if overlap:
        raise ValueError(f"train/val 名单有 {len(overlap)} 个重叠样本，例如: {sorted(overlap)[:5]}")
    unused = len(tokens) - len(selected["train"]) - len(selected["val"])
    if unused:
        print(f"Note: {unused} 个数据集样本不在任何名单里，不会参与训练/验证。")
    return selected["train"], selected["val"]


def main() -> None:
    args = parse_args()
    use_lists = bool(args.train_list or args.val_list)
    if use_lists and not (args.train_list and args.val_list):
        raise ValueError("--train-list 和 --val-list 需同时提供。")
    if not use_lists and abs(args.train_ratio + args.val_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio must equal 1.0")

    output_dir = ensure_dir(args.output_dir)
    train_path = Path(output_dir) / "train.txt"
    val_path = Path(output_dir) / "val.txt"

    if args.recursive:
        root = Path(args.images_dir)
        pairs, unmatched = collect_hierarchical_pairs(root, args.images_subdir, args.masks_subdir)
        if not pairs:
            raise FileNotFoundError(
                f"No matching image/mask pairs found under {root} "
                f"(images subdir='{args.images_subdir}', masks subdir='{args.masks_subdir}')."
            )
        if unmatched:
            print(f"Warning: {len(unmatched)} image(s) had no same-name mask and were skipped, e.g.:")
            for image_path in unmatched[:10]:
                print(f"  {image_path}")

        tokens = [token for token, _, _ in pairs]
        if use_lists:
            train_tokens, val_tokens = split_by_lists(tokens, args.train_list, args.val_list)
        else:
            shuffled = list(tokens)
            random.Random(args.seed).shuffle(shuffled)
            train_count = int(len(shuffled) * args.train_ratio)
            train_tokens = set(shuffled[:train_count])
            val_tokens = set(shuffled) - train_tokens

        # mask_index.json 记录 图片绝对路径 -> mask 绝对路径，训练时据此精确配对(不受同名 stem 影响)
        mask_index = {str(image_path): str(mask_path) for _, image_path, mask_path in pairs}
        mask_index_path = Path(output_dir) / "mask_index.json"
        with mask_index_path.open("w", encoding="utf-8") as file:
            json.dump(mask_index, file, ensure_ascii=False, indent=2)

        def write_tokens(path: Path, keep: set[str]) -> int:
            ordered = [token for token, _, _ in pairs if token in keep]
            with path.open("w", encoding="utf-8") as file:
                for token in ordered:
                    file.write(f"{token}\n")
            return len(ordered)

        n_train = write_tokens(train_path, train_tokens)
        n_val = write_tokens(val_path, val_tokens)
        print(f"Saved train split: {train_path}")
        print(f"Saved val split: {val_path}")
        print(f"Saved mask index: {mask_index_path}")
        print(f"Train samples: {n_train}")
        print(f"Val samples: {n_val}")
        print(f"Note: train 时把 --images-dir 指向数据集根 {root} 即可(mask 由 mask_index.json 定位)。")
        return

    if not args.masks_dir:
        raise ValueError("--masks-dir is required unless --recursive is used.")
    image_by_stem, common_stems = collect_flat_stems(args.images_dir, args.masks_dir)
    if use_lists:
        train_keep, val_keep = split_by_lists(common_stems, args.train_list, args.val_list)
        train_stems = [stem for stem in common_stems if stem in train_keep]
        val_stems = [stem for stem in common_stems if stem in val_keep]
    else:
        random.Random(args.seed).shuffle(common_stems)
        train_count = int(len(common_stems) * args.train_ratio)
        train_stems = common_stems[:train_count]
        val_stems = common_stems[train_count:]

    def write_tokens(path: Path, stems: list[str]) -> None:
        with path.open("w", encoding="utf-8") as file:
            for stem in stems:
                token = image_by_stem[stem].name if args.use_filename else stem
                file.write(f"{token}\n")

    write_tokens(train_path, train_stems)
    write_tokens(val_path, val_stems)

    print(f"Saved train split: {train_path}")
    print(f"Saved val split: {val_path}")
    print(f"Train samples: {len(train_stems)}")
    print(f"Val samples: {len(val_stems)}")


if __name__ == "__main__":
    main()
