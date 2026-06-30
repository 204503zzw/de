import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import segmentation_models_pytorch as smp

from common import (
    compute_mask_iou,
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_gt_mask(gt_dir: Path, stem: str) -> np.ndarray | None:
    for ext in (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".webp"):
        gt_path = gt_dir / f"{stem}{ext}"
        if gt_path.is_file():
            gt = np.asarray(Image.open(gt_path).convert("L"), dtype=np.uint8)
            return gt
    return None


def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    p = pred > 127
    g = gt > 127
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
    valid_ious = [v for v in ious if v == v]
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

    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    if gt_dir is not None and not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    global_tp = 0.0
    global_fp = 0.0
    global_fn = 0.0
    global_tn = 0.0
    smp_global_tp: torch.Tensor | None = None
    smp_global_fp: torch.Tensor | None = None
    smp_global_fn: torch.Tensor | None = None
    smp_global_tn: torch.Tensor | None = None
    per_image_metrics: list[tuple[str, dict[str, float]]] = []
    evaluated_count = 0

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
                pad=pad,
                pad_align=pad_align,
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
                if num_classes == 1:
                    current_metrics = compute_binary_metrics(mask, gt_mask)
                    global_tp += current_metrics["TP"]
                    global_fp += current_metrics["FP"]
                    global_fn += current_metrics["FN"]
                    global_tn += current_metrics["TN"]
                else:
                    current_metrics = compute_multiclass_metrics(mask, gt_mask, num_classes)
                iou_val, s_tp, s_fp, s_fn, s_tn = compute_mask_iou(
                    mask, gt_mask, num_classes,
                )
                if num_classes == 1:
                    current_metrics["IoU"] = iou_val
                else:
                    current_metrics["mIoU"] = iou_val
                if smp_global_tp is None:
                    smp_global_tp = s_tp
                    smp_global_fp = s_fp
                    smp_global_fn = s_fn
                    smp_global_tn = s_tn
                else:
                    smp_global_tp = smp_global_tp + s_tp
                    smp_global_fp = smp_global_fp + s_fp
                    smp_global_fn = smp_global_fn + s_fn
                    smp_global_tn = smp_global_tn + s_tn
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
                num_classes=num_classes,
                alpha=args.overlay_alpha,
            )
            if current_metrics is not None:
                save_overlay(
                    image_path=image_path,
                    mask=mask,
                    output_path=output_dir / f"{stem}_overlay_metrics.png",
                    num_classes=num_classes,
                    alpha=args.overlay_alpha,
                    metrics=current_metrics,
                )
        print(f"Saved results for {image_path}")

    if gt_dir is not None and evaluated_count > 0:
        print(f"\n{'=' * 60}")
        print(f"Evaluation Summary ({evaluated_count} images)")
        print(f"{'=' * 60}")
        summary: dict[str, dict[str, float]] = {}
        if num_classes == 1:
            total = global_tp + global_fp + global_fn + global_tn
            global_iou = float(smp.metrics.iou_score(
                smp_global_tp, smp_global_fp, smp_global_fn, smp_global_tn,
                reduction="micro",
            ).item()) if smp_global_tp is not None else float("nan")
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
        elif smp_global_tp is not None:
            global_iou = float(smp.metrics.iou_score(
                smp_global_tp, smp_global_fp, smp_global_fn, smp_global_tn,
                reduction="micro",
            ).item())
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
