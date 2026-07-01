"""将原始图像和 mask 切成固定大小的 tile，用于切片训练实验。

支持:
- 可配置切片尺寸（256, 384, 512 等）
- 可配置切片重叠比例（0~0.5）
- 自动跳过全背景 tile（可选）
- 同步切割图像和 mask，保持文件名一一对应
- 可选 bottom-right padding 保留边缘区域

用法示例::

    # 无重叠 256x256 切片
    python prepare_tiles.py \\
        --images-dir /data/images --masks-dir /data/masks \\
        --output-dir /data/tiles_256 --tile-size 256

    # 25% 重叠 384x384 切片
    python prepare_tiles.py \\
        --images-dir /data/images --masks-dir /data/masks \\
        --output-dir /data/tiles_384_overlap25 \\
        --tile-size 384 --overlap 0.25

    # 无重叠 512x512，跳过全背景 tile
    python prepare_tiles.py \\
        --images-dir /data/images --masks-dir /data/masks \\
        --output-dir /data/tiles_512 --tile-size 512 --skip-empty
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from common import build_file_maps, ensure_dir, read_split_tokens

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将原始图像和 mask 切成固定大小的 tile"
    )
    parser.add_argument("--images-dir", type=str, required=True,
                        help="原始图像目录")
    parser.add_argument("--masks-dir", type=str, required=True,
                        help="原始 mask 目录")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录，下设 images/ 和 masks/ 子目录")
    parser.add_argument("--tile-size", type=int, default=256,
                        help="切片尺寸（正方形），默认 256")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="切片重叠比例，范围 [0, 0.5)，默认 0（无重叠）")
    parser.add_argument("--skip-empty", action="store_true",
                        help="跳过 mask 全为 0 的 tile（去除纯背景切片）")
    parser.add_argument("--min-foreground-ratio", type=float, default=0.0,
                        help="最小前景像素比例，低于此比例的 tile 被跳过（需配合 --skip-empty）")
    parser.add_argument("--pad", action="store_true",
                        help="边缘不足一个 tile 时用黑色填充（保留边缘区域）")
    parser.add_argument("--split-txt", type=str, default=None,
                        help="可选：只处理指定划分文件中的样本")
    return parser.parse_args()


def compute_tile_positions(
    image_length: int,
    tile_length: int,
    overlap: float,
    pad: bool,
) -> list[int]:
    """计算沿一个维度的切片起始位置列表。"""
    if image_length <= tile_length:
        return [0]

    stride = max(1, int(tile_length * (1.0 - overlap)))
    positions: list[int] = []
    pos = 0
    while pos + tile_length <= image_length:
        positions.append(pos)
        pos += stride

    if pad and positions[-1] + tile_length < image_length:
        positions.append(image_length - tile_length)

    if not pad and not positions:
        positions.append(0)

    return positions


def extract_tiles(
    image: Image.Image,
    mask: Image.Image,
    tile_size: int,
    overlap: float,
    pad: bool,
    skip_empty: bool,
    min_foreground_ratio: float,
) -> list[tuple[Image.Image, Image.Image, int, int]]:
    """从一对 image/mask 中提取所有 tile。

    返回 [(image_tile, mask_tile, row_pos, col_pos), ...]
    """
    img_w, img_h = image.size
    y_positions = compute_tile_positions(img_h, tile_size, overlap, pad)
    x_positions = compute_tile_positions(img_w, tile_size, overlap, pad)

    tiles: list[tuple[Image.Image, Image.Image, int, int]] = []

    for y in y_positions:
        for x in x_positions:
            right = x + tile_size
            bottom = y + tile_size

            if right <= img_w and bottom <= img_h:
                img_tile = image.crop((x, y, right, bottom))
                msk_tile = mask.crop((x, y, right, bottom))
            else:
                crop_right = min(right, img_w)
                crop_bottom = min(bottom, img_h)
                img_crop = image.crop((x, y, crop_right, crop_bottom))
                msk_crop = mask.crop((x, y, crop_right, crop_bottom))

                if pad:
                    img_tile = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                    msk_tile = Image.new("L", (tile_size, tile_size), 0)
                    img_tile.paste(img_crop, (0, 0))
                    msk_tile.paste(msk_crop, (0, 0))
                else:
                    img_tile = img_crop
                    msk_tile = msk_crop

            if skip_empty:
                msk_array = np.asarray(msk_tile, dtype=np.uint8)
                fg_ratio = float(np.sum(msk_array > 0)) / max(msk_array.size, 1)
                if fg_ratio < max(min_foreground_ratio, 1e-8):
                    continue

            tiles.append((img_tile, msk_tile, y, x))

    return tiles


def main() -> None:
    args = parse_args()

    if args.overlap < 0 or args.overlap >= 0.5:
        raise ValueError("overlap 必须在 [0, 0.5) 范围内")

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    out_images_dir = ensure_dir(output_dir / "images")
    out_masks_dir = ensure_dir(output_dir / "masks")

    image_by_name, image_by_stem = build_file_maps(images_dir)
    mask_by_name, mask_by_stem = build_file_maps(masks_dir)

    if args.split_txt:
        stems = read_split_tokens(args.split_txt)
        sample_stems = [s for s in stems if s in image_by_stem and s in mask_by_stem]
    else:
        sample_stems = sorted(set(image_by_stem) & set(mask_by_stem))

    if not sample_stems:
        raise FileNotFoundError("没有找到匹配的 image/mask 对")

    total_tiles = 0
    skipped_empty = 0
    tile_ids: list[str] = []

    print(f"切片参数: tile_size={args.tile_size}, overlap={args.overlap}, "
          f"pad={args.pad}, skip_empty={args.skip_empty}")
    print(f"待处理样本数: {len(sample_stems)}")

    for idx, stem in enumerate(sample_stems):
        image_path = image_by_stem[stem]
        mask_path = mask_by_stem[stem]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if image.size != mask.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)

        tiles = extract_tiles(
            image=image,
            mask=mask,
            tile_size=args.tile_size,
            overlap=args.overlap,
            pad=args.pad,
            skip_empty=args.skip_empty,
            min_foreground_ratio=args.min_foreground_ratio,
        )

        original_tile_count = len(
            compute_tile_positions(image.size[1], args.tile_size, args.overlap, args.pad)
        ) * len(
            compute_tile_positions(image.size[0], args.tile_size, args.overlap, args.pad)
        )
        skipped_empty += original_tile_count - len(tiles)

        for img_tile, msk_tile, row, col in tiles:
            tile_id = f"{stem}_r{row}_c{col}"
            img_tile.save(out_images_dir / f"{tile_id}.png")
            msk_tile.save(out_masks_dir / f"{tile_id}.png")
            tile_ids.append(tile_id)
            total_tiles += 1

        if (idx + 1) % 50 == 0 or idx == len(sample_stems) - 1:
            print(f"  [{idx + 1}/{len(sample_stems)}] 已处理 {stem}, "
                  f"累计 tile 数: {total_tiles}")

    tiles_txt_path = output_dir / "all_tiles.txt"
    with tiles_txt_path.open("w", encoding="utf-8") as f:
        for tile_id in tile_ids:
            f.write(f"{tile_id}\n")

    print(f"\n切片完成:")
    print(f"  输入样本数: {len(sample_stems)}")
    print(f"  输出 tile 数: {total_tiles}")
    if args.skip_empty:
        print(f"  跳过空白 tile 数: {skipped_empty}")
    print(f"  输出目录: {output_dir}")
    print(f"  tile 列表: {tiles_txt_path}")


if __name__ == "__main__":
    main()
