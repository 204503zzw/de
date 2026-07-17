import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def compute_mask_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    threshold: int = 127,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """纯 numpy 实现的 IoU，语义与 smp.metrics.get_stats + micro IoU 一致。

    返回 (iou, tp, fp, fn, tn)，其中 tp/fp/fn/tn 为按类别统计的混淆计数
    （binary 为标量数组，multiclass 为长度 num_classes 的数组），可累加后重新
    计算 Global（micro）IoU。
    """
    if num_classes == 1:
        p = pred > threshold
        g = gt > 0
        tp = np.asarray(np.sum(p & g), dtype=np.int64)
        fp = np.asarray(np.sum(p & ~g), dtype=np.int64)
        fn = np.asarray(np.sum(~p & g), dtype=np.int64)
        tn = np.asarray(np.sum(~p & ~g), dtype=np.int64)
    else:
        p = pred.astype(np.int64)
        g = gt.astype(np.int64)
        tp = np.zeros((num_classes,), dtype=np.int64)
        fp = np.zeros((num_classes,), dtype=np.int64)
        fn = np.zeros((num_classes,), dtype=np.int64)
        tn = np.zeros((num_classes,), dtype=np.int64)
        for c in range(num_classes):
            pc = p == c
            gc = g == c
            tp[c] = np.sum(pc & gc)
            fp[c] = np.sum(pc & ~gc)
            fn[c] = np.sum(~pc & gc)
            tn[c] = np.sum(~pc & ~gc)
    denom = float(tp.sum() + fp.sum() + fn.sum())
    iou = float(tp.sum()) / denom if denom > 0 else float("nan")
    return iou, tp, fp, fn, tn


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


def _draw_metrics_on_image(image: Image.Image, metrics: dict[str, float]) -> Image.Image:
    """在图像左上角绘制精度指标文本（半透明黑底白字）。"""
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()
    lines = []
    for k, v in metrics.items():
        if k in ("TP", "FP", "FN", "TN"):
            continue
        lines.append(f"{k}: {v:.4f}")
    text = "\n".join(lines)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    padding = 6
    draw.rectangle(
        [(0, 0), (text_w + padding * 2, text_h + padding * 2)],
        fill=(0, 0, 0, 180),
    )
    draw.text((padding, padding), text, fill=(255, 255, 255, 255), font=font)
    return image


def save_overlay(
    image_path: str | Path,
    mask: np.ndarray,
    output_path: str | Path,
    num_classes: int = 1,
    color: tuple[int, int, int] = (220, 80, 100),
    alpha: float = 0.45,
    metrics: dict[str, float] | None = None,
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
    result = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    if metrics is not None:
        result = _draw_metrics_on_image(result, metrics)
    result.save(output_path)


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
    parser.add_argument("--dynamic", action="store_true",
                        help="动态推理：保持原图尺寸，仅填充到 stride 的倍数后推理，避免 resize 变形")
    parser.add_argument("--stride", type=int, default=32,
                        help="动态推理时的对齐步长（默认 32，适合 UNet 等 5 层下采样架构）")
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
    parser.add_argument("--gt-dir", type=str, default=None,
                        help="Ground truth mask 目录，用于计算精度指标（IoU、Dice、Precision、Recall 等）")
    parser.add_argument("--metrics-output", type=str, default=None,
                        help="精度指标保存路径（CSV 文件），需配合 --gt-dir 使用")
    return parser.parse_args()


def save_metrics_csv(
    csv_path: str | Path,
    per_image: list[tuple[str, dict[str, float]]],
    summary: dict[str, dict[str, float]],
) -> None:
    """将每张图指标和汇总写入 CSV 文件。"""
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    if not per_image:
        return
    metric_keys = [k for k in per_image[0][1].keys() if k not in ("TP", "FP", "FN", "TN")]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image"] + metric_keys)
        for name, m in per_image:
            row = [name] + [f"{m.get(k, float('nan')):.6f}" for k in metric_keys]
            writer.writerow(row)
        writer.writerow([])
        for label, m in summary.items():
            row = [label] + [f"{m.get(k, float('nan')):.6f}" for k in metric_keys]
            writer.writerow(row)
    print(f"Metrics saved to {csv_path}")


def load_gt_mask(gt_dir: Path, stem: str) -> np.ndarray | None:
    """从 gt_dir 中找到与 stem 同名的 mask 文件并加载为二值数组。"""
    for ext in (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".webp"):
        gt_path = gt_dir / f"{stem}{ext}"
        if gt_path.is_file():
            gt = np.asarray(Image.open(gt_path).convert("L"), dtype=np.uint8)
            return gt
    return None


def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """计算二值分割的精度指标。pred 和 gt 均为 uint8 (0 或 255)。"""
    p = pred > 127
    g = gt > 0
    tp = float(np.sum(p & g))
    fp = float(np.sum(p & ~g))
    fn = float(np.sum(~p & g))
    tn = float(np.sum(~p & ~g))
    total = tp + fp + fn + tn
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    accuracy = (tp + tn) / total if total > 0 else float("nan")
    return {
        "IoU": iou, "Dice": dice, "Precision": precision,
        "Recall": recall, "Accuracy": accuracy,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def compute_multiclass_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> dict[str, float]:
    """计算多类分割的精度指标。pred 和 gt 均为 uint8 类别索引。"""
    total = float(pred.size)
    correct = float(np.sum(pred == gt))
    accuracy = correct / total if total > 0 else float("nan")
    ious = []
    dices = []
    for c in range(num_classes):
        p = pred == c
        g = gt == c
        inter = float(np.sum(p & g))
        union = float(np.sum(p | g))
        iou = inter / union if union > 0 else float("nan")
        dice = 2 * inter / (float(np.sum(p)) + float(np.sum(g))) if (float(np.sum(p)) + float(np.sum(g))) > 0 else float("nan")
        ious.append(iou)
        dices.append(dice)
    valid_ious = [v for v in ious if v == v]  # filter nan
    valid_dices = [v for v in dices if v == v]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else float("nan")
    mean_dice = sum(valid_dices) / len(valid_dices) if valid_dices else float("nan")
    result: dict[str, float] = {"mIoU": mean_iou, "mDice": mean_dice, "Accuracy": accuracy}
    for c in range(num_classes):
        result[f"IoU_c{c}"] = ious[c]
        result[f"Dice_c{c}"] = dices[c]
    return result


def print_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    parts = []
    for k, v in metrics.items():
        if k in ("TP", "FP", "FN", "TN"):
            continue
        parts.append(f"{k}={v:.4f}")
    print(f"{prefix}{' | '.join(parts)}")


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


def _ceil_to_multiple(value: int, multiple: int) -> int:
    """将 value 向上取整到 multiple 的倍数。"""
    return ((value + multiple - 1) // multiple) * multiple


def _pad_to_stride(image: Image.Image, stride: int) -> tuple[Image.Image, dict[str, int]]:
    """将图像右侧和下侧填充到 stride 的倍数，不缩放。"""
    orig_w, orig_h = image.size
    target_w = _ceil_to_multiple(orig_w, stride)
    target_h = _ceil_to_multiple(orig_h, stride)
    if target_w == orig_w and target_h == orig_h:
        pad_info = {"pad_left": 0, "pad_top": 0, "src_left": 0, "src_top": 0,
                    "copy_w": orig_w, "copy_h": orig_h}
        return image, pad_info
    canvas = Image.new(image.mode, (target_w, target_h), 0)
    canvas.paste(image, (0, 0))
    pad_info = {"pad_left": 0, "pad_top": 0, "src_left": 0, "src_top": 0,
                "copy_w": orig_w, "copy_h": orig_h}
    return canvas, pad_info


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
    dynamic: bool = False,
    stride: int = 32,
) -> tuple[np.ndarray, tuple[int, int], dict[str, int] | None]:
    image = Image.open(image_path).convert("L" if channels == 1 else "RGB")
    original_size = image.size
    pad_info: dict[str, int] | None = None
    if dynamic:
        image, pad_info = _pad_to_stride(image, stride)
    elif image_size is not None:
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
    dynamic = bool(args.dynamic)
    stride = int(args.stride)
    pad = bool(args.pad) and image_size is not None
    pad_align = str(args.pad_align or "center")
    if bool(args.pad) and image_size is None and not dynamic:
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
            "dynamic": dynamic,
            "stride": stride,
            "pad": pad,
            "pad_align": pad_align,
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
    )

    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    if gt_dir is not None and not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    global_tp = 0.0
    global_fp = 0.0
    global_fn = 0.0
    global_tn = 0.0
    smp_global_tp = 0.0
    smp_global_fp = 0.0
    smp_global_fn = 0.0
    smp_global_tn = 0.0
    per_image_metrics: list[tuple[str, dict[str, float]]] = []
    evaluated_count = 0

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
            dynamic=dynamic,
            stride=stride,
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

        current_metrics: dict[str, float] | None = None
        if gt_dir is not None:
            gt_mask = load_gt_mask(gt_dir, stem)
            if gt_mask is not None:
                if gt_mask.shape != mask.shape:
                    gt_mask = np.asarray(
                        Image.fromarray(gt_mask).resize(
                            (mask.shape[1], mask.shape[0]), Image.Resampling.NEAREST
                        ),
                        dtype=np.uint8,
                    )
                if int(runtime_num_classes) == 1:
                    current_metrics = compute_binary_metrics(mask, gt_mask)
                    global_tp += current_metrics["TP"]
                    global_fp += current_metrics["FP"]
                    global_fn += current_metrics["FN"]
                    global_tn += current_metrics["TN"]
                else:
                    current_metrics = compute_multiclass_metrics(mask, gt_mask, int(runtime_num_classes))
                iou_val, s_tp, s_fp, s_fn, s_tn = compute_mask_iou(
                    mask, gt_mask, int(runtime_num_classes),
                )
                if int(runtime_num_classes) == 1:
                    current_metrics["IoU"] = iou_val
                else:
                    current_metrics["mIoU"] = iou_val
                smp_global_tp += float(s_tp.sum().item())
                smp_global_fp += float(s_fp.sum().item())
                smp_global_fn += float(s_fn.sum().item())
                smp_global_tn += float(s_tn.sum().item())
                per_image_metrics.append((stem, current_metrics))
                evaluated_count += 1
                print_metrics(current_metrics, prefix=f"[{stem}] ")
            else:
                print(f"[{stem}] Warning: no ground truth found, skipping evaluation")

        if args.overlay:
            save_overlay(
                image_path=image_path,
                mask=mask,
                output_path=output_dir / f"{stem}_overlay.png",
                num_classes=int(runtime_num_classes),
                alpha=args.overlay_alpha,
            )
            if current_metrics is not None:
                save_overlay(
                    image_path=image_path,
                    mask=mask,
                    output_path=output_dir / f"{stem}_overlay_metrics.png",
                    num_classes=int(runtime_num_classes),
                    alpha=args.overlay_alpha,
                    metrics=current_metrics,
                )
        print(f"Saved results for {image_path}")

    if gt_dir is not None and evaluated_count > 0:
        print(f"\n{'=' * 60}")
        print(f"Evaluation Summary ({evaluated_count} images)")
        print(f"{'=' * 60}")
        summary: dict[str, dict[str, float]] = {}
        num_classes_final = int(runtime_num_classes) if 'runtime_num_classes' in dir() else 1
        if num_classes_final == 1:
            total = global_tp + global_fp + global_fn + global_tn
            smp_denom = smp_global_tp + smp_global_fp + smp_global_fn
            global_iou = smp_global_tp / smp_denom if smp_denom > 0 else float("nan")
            global_results = {
                "IoU": global_iou,
                "Dice": 2 * global_tp / (2 * global_tp + global_fp + global_fn) if (2 * global_tp + global_fp + global_fn) > 0 else float("nan"),
                "Precision": global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else float("nan"),
                "Recall": global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else float("nan"),
                "Accuracy": (global_tp + global_tn) / total if total > 0 else float("nan"),
            }
            summary["Global"] = global_results
            print("[Global (pixel-level)]")
            print_metrics(global_results, prefix="  ")
        else:
            smp_denom = smp_global_tp + smp_global_fp + smp_global_fn
            global_iou = smp_global_tp / smp_denom if smp_denom > 0 else float("nan")
            global_results = {"mIoU": global_iou}
            summary["Global"] = global_results
            print("[Global (pixel-level)]")
            print_metrics(global_results, prefix="  ")
        mean_metrics_agg: dict[str, list[float]] = {}
        for _, m in per_image_metrics:
            for k, v in m.items():
                if k in ("TP", "FP", "FN", "TN"):
                    continue
                if v == v:  # not nan
                    mean_metrics_agg.setdefault(k, []).append(v)
        mean_results: dict[str, float] = {}
        print("[Mean (per-image average)]")
        parts = []
        for k, values in mean_metrics_agg.items():
            avg = sum(values) / len(values)
            mean_results[k] = avg
            parts.append(f"{k}={avg:.4f}")
        print(f"  {' | '.join(parts)}")
        print(f"{'=' * 60}")
        summary["Mean"] = mean_results

        if args.metrics_output:
            save_metrics_csv(args.metrics_output, per_image_metrics, summary)


if __name__ == "__main__":
    main()
