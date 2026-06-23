import argparse
from pathlib import Path

import torch

from common import (
    ensure_dir,
    list_input_images,
    load_image_for_inference,
    load_model_from_checkpoint,
    predict_mask_from_logits,
    resize_mask_to_original,
    sahi_predict,
    save_mask,
    save_overlay,
    unpad_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--sahi", action="store_true", default=False,
                        help="启用 SAHI 滑窗推理，适合高分辨率大图")
    parser.add_argument("--sahi-overlap", type=float, default=0.2,
                        help="SAHI 切片重叠比例，范围 [0, 1)，默认 0.2")
    parser.add_argument("--sahi-crop-size", nargs=2, type=int, default=None,
                        metavar=("HEIGHT", "WIDTH"),
                        help="SAHI 从原图裁剪的窗口大小 HEIGHT WIDTH，默认使用训练时的 image_size")
    parser.add_argument("--sahi-model-size", nargs=2, type=int, default=None,
                        metavar=("HEIGHT", "WIDTH"),
                        help="SAHI 裁片 resize 后送入模型的尺寸 HEIGHT WIDTH，默认使用训练时的 image_size")
    parser.add_argument("--overlay", action="store_true", default=False,
                        help="额外保存 mask 叠加在原图上的可视化结果")
    parser.add_argument("--overlay-alpha", type=float, default=0.45,
                        help="叠加透明度，范围 (0, 1)，默认 0.45")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    image_size = tuple(checkpoint["image_size"])
    threshold = float(checkpoint.get("threshold", 0.5)) if args.threshold < 0 else args.threshold
    num_classes = int(checkpoint["model_config"]["num_classes"])
    preprocessing = checkpoint["preprocessing"]
    pad = bool(checkpoint.get("pad", False))
    pad_align = str(checkpoint.get("pad_align", "center") or "center")

    input_images = list_input_images(args.input)
    if not input_images:
        raise FileNotFoundError(f"No input images found in {args.input}")

    for image_path in input_images:
        if args.sahi:
            crop_size = tuple(args.sahi_crop_size) if args.sahi_crop_size else image_size
            model_size = tuple(args.sahi_model_size) if args.sahi_model_size else image_size
            mask, probability = sahi_predict(
                model=model,
                image_path=image_path,
                crop_size=crop_size,
                model_size=model_size,
                preprocessing=preprocessing,
                num_classes=num_classes,
                threshold=threshold,
                overlap_ratio=args.sahi_overlap,
                device=device,
            )
        else:
            image_tensor, original_size, pad_info = load_image_for_inference(
                image_path=image_path,
                image_size=image_size,
                preprocessing=preprocessing,
                pad=pad,
                pad_align=pad_align,
            )
            image_tensor = image_tensor.to(device)

            with torch.inference_mode():
                logits = model(image_tensor)

            mask, probability = predict_mask_from_logits(logits, num_classes, threshold)
            if pad_info is not None:
                mask = unpad_mask(mask, original_size, pad_info)
                probability = unpad_mask(probability, original_size, pad_info)
            else:
                mask = resize_mask_to_original(mask, original_size)
                probability = resize_mask_to_original(probability, original_size)

        stem = Path(image_path).stem
        save_mask(mask, output_dir / f"{stem}_mask.png")
        save_mask(probability, output_dir / f"{stem}_prob.png")
        if args.overlay:
            save_overlay(
                image_path=image_path,
                mask=mask,
                output_path=output_dir / f"{stem}_overlay.png",
                num_classes=num_classes,
                alpha=args.overlay_alpha,
            )
        print(f"Saved results for {image_path}")


if __name__ == "__main__":
    main()
