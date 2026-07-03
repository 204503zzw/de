"""生成 K-fold 交叉验证的训练/验证划分文件。

支持:
- 可配置 K 值（默认 5）
- 可配置随机种子
- 兼容 prepare_splits.py 的输出格式（每行一个样本 id）
- 自动检测 image/mask 匹配

用法示例::

    # 5-fold 交叉验证
    python prepare_kfold_splits.py \\
        --images-dir /data/images --masks-dir /data/masks \\
        --output-dir /data/kfold_5 --k 5

    # 3-fold 交叉验证，指定种子
    python prepare_kfold_splits.py \\
        --images-dir /data/images --masks-dir /data/masks \\
        --output-dir /data/kfold_3 --k 3 --seed 123

输出结构::

    output-dir/
        fold_1/
            train.txt
            val.txt
        fold_2/
            train.txt
            val.txt
        ...
        all_samples.txt
"""

import argparse
import random
from pathlib import Path

from common import build_file_maps, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 K-fold 交叉验证划分文件"
    )
    parser.add_argument("--images-dir", type=str, required=True,
                        help="图像目录")
    parser.add_argument("--masks-dir", type=str, required=True,
                        help="Mask 目录")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--k", type=int, default=5,
                        help="折数（默认 5）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")
    parser.add_argument("--use-filename", action="store_true",
                        help="使用完整文件名（含扩展名）作为样本 id，否则使用 stem")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k = args.k
    if k < 2:
        raise ValueError("K 必须 >= 2")

    image_by_name, image_by_stem = build_file_maps(args.images_dir)
    mask_by_name, mask_by_stem = build_file_maps(args.masks_dir)

    common_stems = sorted(set(image_by_stem) & set(mask_by_stem))
    if not common_stems:
        raise FileNotFoundError("没有找到匹配的 image/mask 对")

    if len(common_stems) < k:
        raise ValueError(f"样本数 ({len(common_stems)}) 小于折数 ({k})，无法划分")

    random.Random(args.seed).shuffle(common_stems)

    output_dir = ensure_dir(args.output_dir)

    all_path = output_dir / "all_samples.txt"
    with all_path.open("w", encoding="utf-8") as f:
        for stem in sorted(common_stems):
            token = image_by_stem[stem].name if args.use_filename else stem
            f.write(f"{token}\n")

    fold_size = len(common_stems) // k
    remainder = len(common_stems) % k

    folds: list[list[str]] = []
    start = 0
    for i in range(k):
        end = start + fold_size + (1 if i < remainder else 0)
        folds.append(common_stems[start:end])
        start = end

    for fold_idx in range(k):
        fold_dir = ensure_dir(output_dir / f"fold_{fold_idx + 1}")
        val_stems = folds[fold_idx]
        train_stems = []
        for j in range(k):
            if j != fold_idx:
                train_stems.extend(folds[j])

        train_path = fold_dir / "train.txt"
        val_path = fold_dir / "val.txt"

        with train_path.open("w", encoding="utf-8") as f:
            for stem in train_stems:
                token = image_by_stem[stem].name if args.use_filename else stem
                f.write(f"{token}\n")

        with val_path.open("w", encoding="utf-8") as f:
            for stem in val_stems:
                token = image_by_stem[stem].name if args.use_filename else stem
                f.write(f"{token}\n")

        print(f"Fold {fold_idx + 1}: train={len(train_stems)}, val={len(val_stems)}")

    print(f"\n总样本数: {len(common_stems)}")
    print(f"K-fold ({k} 折) 划分已保存到: {output_dir}")
    print(f"全部样本列表: {all_path}")


if __name__ == "__main__":
    main()
