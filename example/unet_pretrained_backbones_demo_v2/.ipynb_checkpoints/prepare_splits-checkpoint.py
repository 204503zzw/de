import argparse
import random
from pathlib import Path

from common import build_file_maps, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--masks-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-filename", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if abs(args.train_ratio + args.val_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio must equal 1.0")

    image_by_name, image_by_stem = build_file_maps(args.images_dir)
    mask_by_name, mask_by_stem = build_file_maps(args.masks_dir)

    common_stems = sorted(set(image_by_stem) & set(mask_by_stem))
    if not common_stems:
        raise FileNotFoundError("No matching image/mask pairs found.")

    random.Random(args.seed).shuffle(common_stems)
    train_count = int(len(common_stems) * args.train_ratio)
    train_stems = common_stems[:train_count]
    val_stems = common_stems[train_count:]

    output_dir = ensure_dir(args.output_dir)
    train_path = Path(output_dir) / "train.txt"
    val_path = Path(output_dir) / "val.txt"

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
