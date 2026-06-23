import argparse
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def list_input_images(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    return [
        path
        for path in sorted(input_path.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def save_mask(mask: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    Image.fromarray(mask).save(output_path)


def _compute_edge(fg: np.ndarray, thickness: int = 2) -> np.ndarray:
    eroded = fg.copy()
    for _ in range(thickness):
        padded = np.pad(eroded, 1, mode="constant", constant_values=False)
        eroded = (
            eroded
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return fg & ~eroded


def save_overlay(
    image_path: str | Path,
    mask: np.ndarray,
    output_path: str | Path,
    num_classes: int = 1,
    color: tuple[int, int, int] = (220, 80, 100),
    alpha: float = 0.45,
) -> None:
    _PALETTE = [
        (220,  80, 100),
        ( 80, 160, 220),
        ( 80, 200, 120),
        (220, 180,  60),
        (160,  80, 220),
        ( 60, 200, 200),
        (220, 120,  60),
        (140, 220,  80),
    ]
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    orig = Image.open(image_path).convert("RGB")
    orig_np = np.asarray(orig, dtype=np.float32)
    overlay = orig_np.copy()
    edge_alpha = min(alpha + 0.35, 0.95)
    if num_classes == 1:
        fg = mask > 0
        col = np.array(color, dtype=np.float32)
        for c in range(3):
            overlay[:, :, c] = np.where(fg, orig_np[:, :, c] * (1 - alpha) + col[c] * alpha, orig_np[:, :, c])
        edge = _compute_edge(fg)
        for c in range(3):
            overlay[:, :, c] = np.where(edge, orig_np[:, :, c] * (1 - edge_alpha) + col[c] * edge_alpha, overlay[:, :, c])
    else:
        for cls_id in range(1, num_classes):
            fg = mask == cls_id
            col = np.array(_PALETTE[cls_id % len(_PALETTE)], dtype=np.float32)
            for c in range(3):
                overlay[:, :, c] = np.where(fg, overlay[:, :, c] * (1 - alpha) + col[c] * alpha, overlay[:, :, c])
            edge = _compute_edge(fg)
            for c in range(3):
                overlay[:, :, c] = np.where(edge, overlay[:, :, c] * (1 - edge_alpha) + col[c] * edge_alpha, overlay[:, :, c])
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)


def _shape_dim_to_int(value) -> int | None:
    if isinstance(value, (int, np.integer)):
        resolved = int(value)
        return resolved if resolved > 0 else None
    try:
        text = str(value).strip()
    except Exception:
        return None
    if text.isdigit():
        resolved = int(text)
        return resolved if resolved > 0 else None
    return None


def _normalize_values(values, channels: int, fill_value: float) -> np.ndarray:
    if not values:
        return np.full((channels,), float(fill_value), dtype=np.float32)
    resolved = []
    for item in list(values):
        try:
            resolved.append(float(item))
        except Exception:
            continue
    if not resolved:
        resolved = [float(fill_value)]
    if len(resolved) == 1 and channels > 1:
        resolved = resolved * channels
    elif len(resolved) < channels:
        resolved = resolved + [float(fill_value)] * (channels - len(resolved))
    elif len(resolved) > channels:
        resolved = resolved[:channels]
    return np.asarray(resolved, dtype=np.float32)


def resolve_preprocessing_stats(args: argparse.Namespace, channels: int) -> tuple[np.ndarray, np.ndarray]:
    if args.mean:
        mean = _normalize_values(args.mean, channels=channels, fill_value=0.0)
    elif channels == 3:
        mean = IMAGENET_MEAN.copy()
    else:
        mean = np.zeros((channels,), dtype=np.float32)

    if args.std:
        std = _normalize_values(args.std, channels=channels, fill_value=1.0)
    elif channels == 3:
        std = IMAGENET_STD.copy()
    else:
        std = np.ones((channels,), dtype=np.float32)

    return mean, std


def preprocess_image_array(
    image: np.ndarray,
    channels: int,
    input_space: str,
    input_range: tuple[float, float],
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    processed = image.astype(np.float32)
    if processed.ndim == 2:
        processed = processed[:, :, None]

    if channels == 1 and processed.shape[2] != 1:
        processed = processed[:, :, :1]
    elif channels != 1 and processed.shape[2] == 1:
        processed = np.repeat(processed, channels, axis=2)

    if str(input_space or "RGB").strip().upper() == "BGR" and processed.shape[2] == 3:
        processed = processed[..., ::-1].copy()

    if processed.size and processed.max() > 1.0 and float(input_range[1]) == 1.0:
        processed = processed / 255.0

    safe_std = np.where(std == 0, 1.0, std)
    processed = processed - mean.reshape((1, 1, -1))
    processed = processed / safe_std.reshape((1, 1, -1))
    return processed.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--imgsz", nargs=2, type=int, default=None)
    parser.add_argument("--pad", action="store_true",
                        help="不缩放，直接把原图居中放到模型尺寸画布上、不足处用黑色填充（需配合 --imgsz 或固定输入尺寸的模型）")
    parser.add_argument("--pad-align", type=str, default="center",
                        choices=["center", "top_left"],
                        help="填充时原图的对齐方式：center 居中，top_left 放在左上角")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--input-space", type=str, default="RGB")
    parser.add_argument("--input-range", nargs=2, type=float, default=[0.0, 1.0])
    parser.add_argument("--mean", nargs="*", type=float, default=None)
    parser.add_argument("--std", nargs="*", type=float, default=None)
    parser.add_argument("--save-prob", action="store_true", default=False)
    parser.add_argument("--overlay", action="store_true", default=False,
                        help="额外保存 mask 叠加在原图上的可视化结果")
    parser.add_argument("--overlay-alpha", type=float, default=0.45,
                        help="叠加透明度，范围 (0, 1)，默认 0.45")
    return parser.parse_args()


def resolve_default_providers(ort) -> list[str]:
    available = list(ort.get_available_providers())
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if providers:
        return providers
    if available:
        return available
    return ["CPUExecutionProvider"]


def resolve_session_config(session, args: argparse.Namespace) -> tuple[str, tuple[int, int] | None, int, int | None, np.ndarray, np.ndarray]:
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_shape = list(input_meta.shape or [])
    output_shape = list(output_meta.shape or [])

    model_channels = _shape_dim_to_int(input_shape[1]) if len(input_shape) >= 2 else None
    model_height = _shape_dim_to_int(input_shape[-2]) if len(input_shape) >= 4 else None
    model_width = _shape_dim_to_int(input_shape[-1]) if len(input_shape) >= 4 else None
    output_channels = _shape_dim_to_int(output_shape[1]) if len(output_shape) >= 2 else None

    image_size = None
    if isinstance(args.imgsz, (list, tuple)) and len(args.imgsz) == 2:
        image_size = (int(args.imgsz[0]), int(args.imgsz[1]))
    elif model_height is not None and model_width is not None:
        image_size = (model_height, model_width)

    channels = int(model_channels or 3)
    mean, std = resolve_preprocessing_stats(args, channels)
    return input_meta.name, image_size, channels, output_channels, mean, std


def prepare_input(
    image_path: str | Path,
    image_size: tuple[int, int] | None,
    channels: int,
    input_space: str,
    input_range: tuple[float, float],
    mean: np.ndarray,
    std: np.ndarray,
    pad: bool = False,
    pad_align: str = "center",
) -> tuple[np.ndarray, tuple[int, int], dict[str, int] | None]:
    image = Image.open(image_path).convert("L" if channels == 1 else "RGB")
    original_size = image.size
    pad_info: dict[str, int] | None = None
    if image_size is not None:
        if pad:
            image, pad_info = pad_image(image, image_size, fill=0, align=pad_align)
        else:
            image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = preprocess_image_array(
        image_array,
        channels=channels,
        input_space=input_space,
        input_range=input_range,
        mean=mean,
        std=std,
    )
    if image_array.ndim == 2:
        image_array = image_array[:, :, None]
    input_tensor = np.transpose(image_array, (2, 0, 1))[None, ...].astype(np.float32)
    return input_tensor, original_size, pad_info


def resize_mask(mask: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(mask).resize(original_size, Image.Resampling.NEAREST))


def pad_image(
    image: Image.Image,
    size: tuple[int, int],
    fill: int = 0,
    align: str = "center",
) -> tuple[Image.Image, dict[str, int]]:
    """不缩放，直接将原图放到目标尺寸 (height, width) 的画布上，不足处填 fill。

    若原图某一边比目标大，则在该边裁剪以放入画布。

    Args:
        align: "center" 居中放置，"top_left" 放在左上角
    """
    target_h, target_w = int(size[0]), int(size[1])
    orig_w, orig_h = image.size
    copy_w = min(orig_w, target_w)
    copy_h = min(orig_h, target_h)
    if align == "top_left":
        src_left = 0
        src_top = 0
        pad_left = 0
        pad_top = 0
    else:
        src_left = (orig_w - copy_w) // 2
        src_top = (orig_h - copy_h) // 2
        pad_left = (target_w - copy_w) // 2
        pad_top = (target_h - copy_h) // 2
    cropped = image.crop((src_left, src_top, src_left + copy_w, src_top + copy_h))
    canvas = Image.new(image.mode, (target_w, target_h), fill)
    canvas.paste(cropped, (pad_left, pad_top))
    pad_info = {
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "src_left": int(src_left),
        "src_top": int(src_top),
        "copy_w": int(copy_w),
        "copy_h": int(copy_h),
    }
    return canvas, pad_info


def unpad_mask(
    mask: np.ndarray,
    original_size: tuple[int, int],
    pad_info: dict[str, int],
) -> np.ndarray:
    """去除填充并还原到原图尺寸 (width, height)，不缩放；被裁掉的边界保持 0。"""
    orig_w, orig_h = int(original_size[0]), int(original_size[1])
    pad_left = int(pad_info["pad_left"])
    pad_top = int(pad_info["pad_top"])
    src_left = int(pad_info["src_left"])
    src_top = int(pad_info["src_top"])
    copy_w = int(pad_info["copy_w"])
    copy_h = int(pad_info["copy_h"])
    valid = mask[pad_top : pad_top + copy_h, pad_left : pad_left + copy_w]
    restored = np.zeros((orig_h, orig_w), dtype=mask.dtype)
    restored[src_top : src_top + copy_h, src_left : src_left + copy_w] = valid
    return restored


def postprocess_output(logits: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, int]:
    scores = np.asarray(logits)
    if scores.ndim == 4:
        scores = scores[0]
    elif scores.ndim == 2:
        scores = scores[None, ...]
    if scores.ndim != 3:
        raise RuntimeError(f"Unexpected output shape: {getattr(logits, 'shape', None)}")

    channel_count = int(scores.shape[0]) if scores.shape[0] > 0 else 1
    if channel_count == 1:
        probabilities = 1.0 / (1.0 + np.exp(-scores[0]))
        mask = (probabilities > float(threshold)).astype(np.uint8) * 255
        probability = np.clip(probabilities * 255.0, 0, 255).astype(np.uint8)
        return mask, probability, channel_count

    scores = scores - scores.max(axis=0, keepdims=True)
    probabilities = np.exp(scores)
    probabilities = probabilities / probabilities.sum(axis=0, keepdims=True)
    mask = np.argmax(probabilities, axis=0).astype(np.uint8)
    probability = np.clip(np.max(probabilities, axis=0) * 255.0, 0, 255).astype(np.uint8)
    return mask, probability, channel_count


def main() -> None:
    args = parse_args()

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError("Please install onnxruntime before running this script.") from error

    providers = resolve_default_providers(ort)
    output_dir = ensure_dir(args.output_dir)
    session = ort.InferenceSession(str(args.onnx), providers=providers)
    input_name, image_size, channels, output_channels, mean, std = resolve_session_config(session, args)
    input_space = str(args.input_space or "RGB").strip().upper()
    input_range = (float(args.input_range[0]), float(args.input_range[1]))
    pad = bool(args.pad) and image_size is not None
    pad_align = str(args.pad_align or "center")
    if bool(args.pad) and image_size is None:
        print("Warning: --pad ignored because input size is dynamic; pass --imgsz to enable padding.")

    input_images = list_input_images(args.input)
    if not input_images:
        raise FileNotFoundError(f"No input images found in {args.input}")

    print(
        "Resolved config:",
        {
            "providers": providers,
            "image_size": list(image_size) if image_size is not None else "original",
            "channels": channels,
            "num_classes": int(output_channels) if output_channels is not None else "auto_from_output",
            "threshold": float(args.threshold),
            "input_space": input_space,
            "input_range": list(input_range),
            "pad": pad,
            "pad_align": pad_align,
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
    )

    for image_path in input_images:
        input_tensor, original_size, pad_info = prepare_input(
            image_path=image_path,
            image_size=image_size,
            channels=channels,
            input_space=input_space,
            input_range=input_range,
            mean=mean,
            std=std,
            pad=pad,
            pad_align=pad_align,
        )
        logits = session.run(None, {input_name: input_tensor})[0]
        mask, probability, runtime_num_classes = postprocess_output(logits, float(args.threshold))
        if pad_info is not None:
            mask = unpad_mask(mask, original_size, pad_info)
            probability = unpad_mask(probability, original_size, pad_info)
        else:
            mask = resize_mask(mask, original_size)
            probability = resize_mask(probability, original_size)

        stem = Path(image_path).stem
        save_mask(mask, output_dir / f"{stem}_mask.png")
        if args.save_prob or int(runtime_num_classes) == 1:
            save_mask(probability, output_dir / f"{stem}_prob.png")
        if args.overlay:
            save_overlay(
                image_path=image_path,
                mask=mask,
                output_path=output_dir / f"{stem}_overlay.png",
                num_classes=int(runtime_num_classes),
                alpha=args.overlay_alpha,
            )
        print(f"Saved results for {image_path}")


if __name__ == "__main__":
    main()
