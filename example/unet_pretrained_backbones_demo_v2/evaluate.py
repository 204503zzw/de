"""独立验证模块 — 加载模型 + 验证集，计算精度指标。

用法与训练时的验证类似，但不需要训练，只做前向推理 + 精度评估。

用法示例::

    # PyTorch checkpoint
    python evaluate.py --checkpoint /path/to/best.pth \
        --images-dir /path/to/images --masks-dir /path/to/masks \
        --val-txt /path/to/val.txt --output-dir /path/to/eval_output

    # ONNX 模型
    python evaluate.py --onnx /path/to/model.onnx \
        --images-dir /path/to/images --masks-dir /path/to/masks \
        --val-txt /path/to/val.txt --output-dir /path/to/eval_output \
        --num-classes 1 --imgsz 640 640

    # 动态推理模式（保持原图尺寸，逐张验证）
    python evaluate.py --checkpoint /path/to/best.pth \
        --images-dir /path/to/images --masks-dir /path/to/masks \
        --val-txt /path/to/val.txt --output-dir /path/to/eval_output \
        --dynamic
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import EvalTransform
from common import (
    SegmentationTxtDataset,
    build_model,
    collect_split_pairs,
    compute_batch_iou,
    create_loss_function,
    ensure_dir,
    get_preprocessing_config,
    load_checkpoint,
    load_optional_unet_meta,
    normalize_arch_name,
    preprocess_image_array,
    resolve_encoder_weights,
    save_json,
    save_overlay,
)


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """二值分割指标（TP/FP/FN/TN 计数 + IoU/Dice/Precision/Recall/Accuracy）。"""
    p = pred > 0
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
    """多类分割指标（mIoU/mDice/Accuracy + 每个类别的 IoU/Dice）。"""
    total = float(pred.size)
    correct = float(np.sum(pred == gt))
    accuracy = correct / total if total > 0 else float("nan")
    ious: list[float] = []
    dices: list[float] = []
    for c in range(num_classes):
        p = pred == c
        g = gt == c
        inter = float(np.sum(p & g))
        union = float(np.sum(p | g))
        iou = inter / union if union > 0 else float("nan")
        denom = float(np.sum(p)) + float(np.sum(g))
        dice = 2 * inter / denom if denom > 0 else float("nan")
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


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    parts = []
    for k, v in metrics.items():
        if k in ("TP", "FP", "FN", "TN"):
            continue
        parts.append(f"{k}={v:.4f}")
    print(f"{prefix}{' | '.join(parts)}")


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


# ---------------------------------------------------------------------------
# 动态推理辅助
# ---------------------------------------------------------------------------

def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pad_to_stride(image: Image.Image, stride: int) -> tuple[Image.Image, tuple[int, int]]:
    """将图像右侧和下侧填充到 stride 的倍数，返回 (填充后图像, 原始尺寸(w,h))。"""
    orig_w, orig_h = image.size
    target_w = _ceil_to_multiple(orig_w, stride)
    target_h = _ceil_to_multiple(orig_h, stride)
    if target_w == orig_w and target_h == orig_h:
        return image, (orig_w, orig_h)
    canvas = Image.new(image.mode, (target_w, target_h), 0)
    canvas.paste(image, (0, 0))
    return canvas, (orig_w, orig_h)


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def load_pytorch_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]
    model = build_model(
        arch=model_config["arch"],
        encoder_name=model_config["encoder_name"],
        encoder_weights=None,
        in_channels=int(model_config["in_channels"]),
        num_classes=int(model_config["num_classes"]),
        encoder_depth=int(model_config.get("encoder_depth", 5)),
        encoder_output_stride=int(model_config.get("encoder_output_stride", 16)),
        decoder_channels=tuple(model_config.get("decoder_channels", [])) or None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def load_onnx_session(onnx_path: str | Path) -> Any:
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    return session


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

def predict_batch_pytorch(
    model: torch.nn.Module,
    images: torch.Tensor,
    num_classes: int,
    threshold: float,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    """PyTorch 批量前向推理，返回 (预测 mask (B,H,W), logits tensor)。"""
    images = images.to(device, non_blocking=True)
    with torch.inference_mode():
        logits = model(images)
    if num_classes == 1:
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs > threshold).cpu().numpy().astype(np.uint8)
    else:
        preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
    return preds, logits


def predict_batch_onnx(
    session: Any,
    images: torch.Tensor,
    num_classes: int,
    threshold: float,
) -> np.ndarray:
    """ONNX 批量前向推理，返回 (B, H, W) 的预测 mask。"""
    input_name = session.get_inputs()[0].name
    input_array = images.numpy().astype(np.float32)
    outputs = session.run(None, {input_name: input_array})
    logits = outputs[0]
    if num_classes == 1:
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = probs.squeeze(1)
        preds = (probs > threshold).astype(np.uint8)
    else:
        preds = np.argmax(logits, axis=1).astype(np.uint8)
    return preds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="独立验证模块 — 加载模型 + 验证集，计算精度指标",
    )
    # 模型（二选一）
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--checkpoint", type=str, default=None,
                             help="PyTorch checkpoint 路径（.pth）")
    model_group.add_argument("--onnx", type=str, default=None,
                             help="ONNX 模型路径（.onnx）")

    # 数据
    parser.add_argument("--images-dir", type=str, required=True,
                        help="图像目录")
    parser.add_argument("--masks-dir", type=str, required=True,
                        help="Ground truth mask 目录")
    parser.add_argument("--val-txt", type=str, required=True,
                        help="验证集划分文件（每行一个样本 id）")

    # 模型参数（ONNX 时需手动指定）
    parser.add_argument("--num-classes", type=int, default=None,
                        help="类别数，PyTorch 从 checkpoint 自动读取，ONNX 需指定（默认 1）")
    parser.add_argument("--imgsz", nargs=2, type=int, default=None,
                        metavar=("HEIGHT", "WIDTH"),
                        help="输入图像尺寸 HEIGHT WIDTH，PyTorch 从 checkpoint 自动读取，ONNX 需指定")
    parser.add_argument("--encoder-name", type=str, default="resnet18",
                        help="编码器名称，ONNX 模式下用于推断预处理参数（默认 resnet18）")
    parser.add_argument("--encoder-weights", type=str, default="imagenet",
                        help="编码器预训练权重名称，用于推断预处理参数（默认 imagenet）")

    # 推理参数
    parser.add_argument("--threshold", type=float, default=-1.0,
                        help="二值分割阈值，-1 表示从 checkpoint 读取（默认 0.5）")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="验证 batch size（默认 8）")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="DataLoader 工作进程数（默认 1）")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备，auto/cpu/cuda（默认 auto）")
    parser.add_argument("--amp", action="store_true",
                        help="启用混合精度推理")
    parser.add_argument("--pad", action="store_true", default=None,
                        help="使用填充模式（PyTorch 从 checkpoint 自动读取，ONNX 需手动指定）")
    parser.add_argument("--pad-align", type=str, default=None,
                        choices=["center", "top_left"],
                        help="填充对齐方式（PyTorch 从 checkpoint 自动读取）")
    parser.add_argument("--dynamic", action="store_true",
                        help="动态推理：保持原图尺寸，仅填充到 stride 的倍数后推理（逐张处理，无 batch）")
    parser.add_argument("--stride", type=int, default=32,
                        help="动态推理时的对齐步长（默认 32）")

    # 输出
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录，保存 overlay 可视化和 CSV 等结果")
    parser.add_argument("--overlay", action="store_true", default=False,
                        help="保存每张图的 overlay 可视化（需指定 --output-dir）")
    parser.add_argument("--overlay-alpha", type=float, default=0.45,
                        help="叠加透明度（默认 0.45）")
    parser.add_argument("--metrics-output", type=str, default=None,
                        help="精度指标 CSV 保存路径（默认保存到 output-dir/metrics.csv）")
    parser.add_argument("--save-masks", action="store_true", default=False,
                        help="保存预测 mask 到 output-dir（需指定 --output-dir）")

    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    name = str(device_name or "").strip().lower()
    if not name or name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    # ------------------------------------------------------------------
    # 加载模型 & 解析配置
    # ------------------------------------------------------------------
    onnx_session = None
    pytorch_model = None

    if args.checkpoint:
        pytorch_model, checkpoint = load_pytorch_model(args.checkpoint, device)
        image_size = tuple(checkpoint["image_size"])
        threshold = float(checkpoint.get("threshold", 0.5)) if args.threshold < 0 else args.threshold
        num_classes = int(checkpoint["model_config"]["num_classes"])
        preprocessing = checkpoint["preprocessing"]
        pad = bool(checkpoint.get("pad", False)) if args.pad is None else args.pad
        pad_align = str(checkpoint.get("pad_align", "center") or "center") if args.pad_align is None else args.pad_align
        mask_values_raw = checkpoint.get("mask_values")
        mask_values = [int(x) for x in list(mask_values_raw or [])] if mask_values_raw else []
        print(f"Loaded PyTorch checkpoint: {args.checkpoint}")
        print(f"  arch={checkpoint['model_config']['arch']} encoder={checkpoint['model_config']['encoder_name']}")
        print(f"  image_size={list(image_size)} num_classes={num_classes} threshold={threshold:.4f}")
        print(f"  pad={pad} pad_align={pad_align}")
    else:
        onnx_session = load_onnx_session(args.onnx)
        if args.imgsz is None:
            input_shape = onnx_session.get_inputs()[0].shape
            if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                image_size = (input_shape[2], input_shape[3])
            elif args.dynamic:
                image_size = None  # type: ignore[assignment]
            else:
                raise ValueError("ONNX 模型输入尺寸为动态，请用 --imgsz H W 指定或使用 --dynamic")
        else:
            image_size = tuple(args.imgsz)
        num_classes = args.num_classes if args.num_classes is not None else 1
        threshold = 0.5 if args.threshold < 0 else args.threshold
        encoder_weights = resolve_encoder_weights(args.encoder_name, args.encoder_weights)
        preprocessing = get_preprocessing_config(args.encoder_name, encoder_weights)
        pad = args.pad if args.pad is not None else False
        pad_align = args.pad_align if args.pad_align is not None else "center"
        mask_values = []
        print(f"Loaded ONNX model: {args.onnx}")
        print(f"  image_size={list(image_size) if image_size else 'dynamic'} num_classes={num_classes} threshold={threshold:.4f}")
        print(f"  pad={pad} pad_align={pad_align}")

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    val_txt = Path(args.val_txt)

    unet_meta = load_optional_unet_meta(val_txt)
    meta_mask_values = unet_meta.get("mask_values") if isinstance(unet_meta, dict) else None
    if meta_mask_values and not mask_values:
        mask_values = [int(x) for x in list(meta_mask_values)]

    dynamic = bool(args.dynamic)
    stride = int(args.stride)

    # ------------------------------------------------------------------
    # 输出目录
    # ------------------------------------------------------------------
    output_dir = ensure_dir(args.output_dir) if args.output_dir else None
    metrics_csv_path = args.metrics_output
    if metrics_csv_path is None and output_dir is not None:
        metrics_csv_path = str(output_dir / "metrics.csv")

    # ------------------------------------------------------------------
    # 公共统计变量
    # ------------------------------------------------------------------
    loss_fn = create_loss_function(num_classes) if pytorch_model is not None else None

    global_tp = 0.0
    global_fp = 0.0
    global_fn = 0.0
    global_tn = 0.0
    per_image_metrics: list[tuple[str, dict[str, float]]] = []
    total_loss_sum = 0.0
    total_iou_sum = 0.0
    batch_count = 0
    started_at = time.time()

    if dynamic:
        # ==============================================================
        # 动态推理模式：逐张处理，保持原图尺寸
        # ==============================================================
        sample_pairs = collect_split_pairs(images_dir, masks_dir, val_txt)
        print(f"\nValidation dataset: {len(sample_pairs)} samples (dynamic mode, stride={stride})")
        print(f"Device: {device}, AMP: {use_amp}")

        for idx, (image_path, mask_path) in enumerate(sample_pairs):
            stem = image_path.stem

            # 加载原图 + GT mask
            image = Image.open(image_path).convert("RGB")
            gt_pil = Image.open(mask_path).convert("L")
            orig_w, orig_h = image.size

            # GT mask 处理
            gt_array = np.asarray(gt_pil, dtype=np.uint8)
            if gt_array.shape != (orig_h, orig_w):
                gt_pil = gt_pil.resize((orig_w, orig_h), Image.Resampling.NEAREST)
                gt_array = np.asarray(gt_pil, dtype=np.uint8)

            # 图像填充到 stride 倍数
            padded_image, (pw, ph) = _pad_to_stride(image, stride)

            # 预处理
            img_array = np.asarray(padded_image, dtype=np.float32)
            img_array = preprocess_image_array(img_array, preprocessing)
            input_tensor = torch.from_numpy(np.transpose(img_array, (2, 0, 1))).float().unsqueeze(0)

            # 推理
            if pytorch_model is not None:
                input_device = input_tensor.to(device, non_blocking=True)
                with torch.inference_mode():
                    autocast_enabled = use_amp and device.type == "cuda"
                    with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                        logits = pytorch_model(input_device)
                if num_classes == 1:
                    probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
                    pred_full = (probs > threshold).astype(np.uint8)
                else:
                    pred_full = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            else:
                input_name = onnx_session.get_inputs()[0].name
                outputs = onnx_session.run(None, {input_name: input_tensor.numpy()})
                raw_logits = outputs[0]
                if num_classes == 1:
                    probs = 1.0 / (1.0 + np.exp(-raw_logits))
                    probs = probs.squeeze(0).squeeze(0)
                    pred_full = (probs > threshold).astype(np.uint8)
                else:
                    pred_full = np.argmax(raw_logits, axis=1).squeeze(0).astype(np.uint8)

            # 裁回原图尺寸
            pred_mask = pred_full[:orig_h, :orig_w]

            # 计算 loss/IoU（PyTorch 模式，需构造对应的 GT tensor）
            if pytorch_model is not None and loss_fn is not None:
                gt_padded = np.zeros((padded_image.size[1], padded_image.size[0]), dtype=np.uint8)
                gt_padded[:orig_h, :orig_w] = gt_array
                if num_classes == 1:
                    gt_tensor = torch.from_numpy((gt_padded > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                else:
                    if mask_values:
                        indexed = np.zeros(gt_padded.shape, dtype=np.int64)
                        for ci, mv in enumerate(mask_values):
                            indexed[gt_padded == mv] = ci
                        gt_tensor = torch.from_numpy(indexed).unsqueeze(0).long().to(device)
                    else:
                        gt_tensor = torch.from_numpy(gt_padded.astype(np.int64)).unsqueeze(0).long().to(device)
                with torch.inference_mode():
                    autocast_enabled = use_amp and device.type == "cuda"
                    with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                        batch_loss = float(loss_fn(logits, gt_tensor).item())
                batch_iou = compute_batch_iou(logits.detach(), gt_tensor.detach(), num_classes, threshold)
                total_loss_sum += batch_loss
                total_iou_sum += batch_iou
                batch_count += 1

            # 逐图指标
            if num_classes == 1:
                gt_binary = (gt_array > 127).astype(np.uint8)
                current_metrics = compute_binary_metrics(pred_mask, gt_binary)
                global_tp += current_metrics["TP"]
                global_fp += current_metrics["FP"]
                global_fn += current_metrics["FN"]
                global_tn += current_metrics["TN"]
            else:
                if mask_values:
                    gt_indexed = np.zeros(gt_array.shape, dtype=np.uint8)
                    for ci, mv in enumerate(mask_values):
                        gt_indexed[gt_array == mv] = ci
                    current_metrics = compute_multiclass_metrics(pred_mask, gt_indexed, num_classes)
                else:
                    current_metrics = compute_multiclass_metrics(pred_mask, gt_array, num_classes)

            per_image_metrics.append((stem, current_metrics))
            print_metrics(current_metrics, prefix=f"[{stem}] ")

            # 保存 overlay
            if args.overlay and output_dir is not None:
                overlay_mask = (pred_mask * 255).astype(np.uint8) if num_classes == 1 else pred_mask
                save_overlay(
                    image_path=image_path, mask=overlay_mask,
                    output_path=output_dir / f"{stem}_overlay.png",
                    num_classes=num_classes, alpha=args.overlay_alpha,
                )
                save_overlay(
                    image_path=image_path, mask=overlay_mask,
                    output_path=output_dir / f"{stem}_overlay_metrics.png",
                    num_classes=num_classes, alpha=args.overlay_alpha,
                    metrics=current_metrics,
                )

            # 保存预测 mask
            if args.save_masks and output_dir is not None:
                mask_out = (pred_mask * 255).astype(np.uint8) if num_classes == 1 else pred_mask
                Image.fromarray(mask_out).save(output_dir / f"{stem}_mask.png")

    else:
        # ==============================================================
        # 固定尺寸模式：使用 DataLoader 批量验证
        # ==============================================================
        val_transform = EvalTransform(image_size=image_size, pad=pad, pad_align=pad_align)

        val_dataset = SegmentationTxtDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            split_txt=val_txt,
            image_size=image_size,
            num_classes=num_classes,
            preprocessing=preprocessing,
            mask_values=mask_values,
            transform=val_transform,
            pad=pad,
            pad_align=pad_align,
        )

        pin_memory = device.type == "cuda"
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=args.num_workers > 0,
        )

        print(f"\nValidation dataset: {len(val_dataset)} samples, {len(val_loader)} batches")
        print(f"Device: {device}, AMP: {use_amp}")

        try:
            enable_live_progress = bool(
                getattr(getattr(__import__("sys"), "stderr", None), "isatty", lambda: False)()
            )
        except Exception:
            enable_live_progress = False

        sample_index = 0
        iterator = tqdm(val_loader, total=len(val_loader), leave=True, disable=not enable_live_progress)

        for images, masks in iterator:
            current_batch_size = images.shape[0]

            # 前向推理
            if pytorch_model is not None:
                preds, logits = predict_batch_pytorch(pytorch_model, images, num_classes, threshold, device)
                masks_device = masks.to(device, non_blocking=True)
                with torch.inference_mode():
                    autocast_enabled = use_amp and device.type == "cuda"
                    with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                        batch_loss = float(loss_fn(logits, masks_device).item())
                batch_iou = compute_batch_iou(logits.detach(), masks_device.detach(), num_classes, threshold)
                total_loss_sum += batch_loss
                total_iou_sum += batch_iou
                batch_count += 1
            else:
                preds = predict_batch_onnx(onnx_session, images, num_classes, threshold)

            # 逐样本计算指标
            for i in range(current_batch_size):
                if sample_index >= len(val_dataset.samples):
                    break

                image_path, mask_path = val_dataset.samples[sample_index]
                stem = image_path.stem

                pred_mask = preds[i]
                if num_classes == 1:
                    gt_mask = masks[i].squeeze(0).numpy()
                    gt_mask = (gt_mask > 0).astype(np.uint8)
                    pred_binary = pred_mask.astype(np.uint8)
                    current_metrics = compute_binary_metrics(pred_binary, gt_mask)
                    global_tp += current_metrics["TP"]
                    global_fp += current_metrics["FP"]
                    global_fn += current_metrics["FN"]
                    global_tn += current_metrics["TN"]
                else:
                    gt_mask = masks[i].numpy().astype(np.uint8)
                    current_metrics = compute_multiclass_metrics(pred_mask, gt_mask, num_classes)

                per_image_metrics.append((stem, current_metrics))
                print_metrics(current_metrics, prefix=f"[{stem}] ")

                # 保存 overlay 可视化
                if args.overlay and output_dir is not None:
                    orig_image = Image.open(image_path).convert("RGB")
                    orig_w, orig_h = orig_image.size

                    if num_classes == 1:
                        overlay_mask = (pred_mask * 255).astype(np.uint8)
                    else:
                        overlay_mask = pred_mask

                    if overlay_mask.shape[0] != orig_h or overlay_mask.shape[1] != orig_w:
                        overlay_mask = np.asarray(
                            Image.fromarray(overlay_mask).resize(
                                (orig_w, orig_h), Image.Resampling.NEAREST
                            ),
                            dtype=np.uint8,
                        )

                    save_overlay(
                        image_path=image_path, mask=overlay_mask,
                        output_path=output_dir / f"{stem}_overlay.png",
                        num_classes=num_classes, alpha=args.overlay_alpha,
                    )
                    save_overlay(
                        image_path=image_path, mask=overlay_mask,
                        output_path=output_dir / f"{stem}_overlay_metrics.png",
                        num_classes=num_classes, alpha=args.overlay_alpha,
                        metrics=current_metrics,
                    )

                # 保存预测 mask
                if args.save_masks and output_dir is not None:
                    mask_out = (pred_mask * 255).astype(np.uint8) if num_classes == 1 else pred_mask
                    Image.fromarray(mask_out).save(output_dir / f"{stem}_mask.png")

                sample_index += 1

            # 更新进度条
            if pytorch_model is not None and batch_count > 0:
                avg_loss = total_loss_sum / batch_count
                avg_iou = total_iou_sum / batch_count
                iterator.set_postfix(loss=f"{avg_loss:.4f}", IoU=f"{avg_iou:.4f}")
            elif per_image_metrics:
                last_metrics = per_image_metrics[-1][1]
                display_iou = last_metrics.get("IoU", last_metrics.get("mIoU", 0.0))
                iterator.set_postfix(IoU=f"{display_iou:.4f}")

    elapsed = time.time() - started_at
    evaluated_count = len(per_image_metrics)

    # ------------------------------------------------------------------
    # 汇总输出
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Evaluation Summary ({evaluated_count} images, {elapsed:.1f}s)")
    print(f"{'=' * 60}")

    # 输出 val_loss 和 val_iou（与训练日志格式一致）
    if pytorch_model is not None and batch_count > 0:
        val_loss = total_loss_sum / batch_count
        val_iou = total_iou_sum / batch_count
        print(f"val_loss={val_loss:.4f}  val_iou={val_iou:.4f}")
    else:
        val_loss = float("nan")
        val_iou = float("nan")

    summary: dict[str, dict[str, float]] = {}

    if num_classes == 1 and evaluated_count > 0:
        total = global_tp + global_fp + global_fn + global_tn
        global_results = {
            "IoU": global_tp / (global_tp + global_fp + global_fn) if (global_tp + global_fp + global_fn) > 0 else float("nan"),
            "Dice": 2 * global_tp / (2 * global_tp + global_fp + global_fn) if (2 * global_tp + global_fp + global_fn) > 0 else float("nan"),
            "Precision": global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else float("nan"),
            "Recall": global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else float("nan"),
            "Accuracy": (global_tp + global_tn) / total if total > 0 else float("nan"),
        }
        summary["Global"] = global_results
        print("[Global (pixel-level)]")
        print_metrics(global_results, prefix="  ")

    if evaluated_count > 0:
        mean_metrics_agg: dict[str, list[float]] = {}
        for _, m in per_image_metrics:
            for k, v in m.items():
                if k in ("TP", "FP", "FN", "TN"):
                    continue
                if v == v:  # not nan
                    mean_metrics_agg.setdefault(k, []).append(v)
        mean_results: dict[str, float] = {}
        for k, values in mean_metrics_agg.items():
            mean_results[k] = sum(values) / len(values)
        summary["Mean"] = mean_results
        print("[Mean (per-image average)]")
        print_metrics(mean_results, prefix="  ")

    print(f"{'=' * 60}")

    # 保存 CSV
    if metrics_csv_path and evaluated_count > 0:
        save_metrics_csv(metrics_csv_path, per_image_metrics, summary)

    # 保存 JSON 汇总
    if output_dir is not None:
        eval_summary: dict[str, Any] = {
            "evaluated_count": evaluated_count,
            "elapsed_seconds": elapsed,
            "num_classes": num_classes,
            "threshold": threshold,
            "dynamic": dynamic,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "summary": {k: {mk: float(mv) for mk, mv in v.items()} for k, v in summary.items()},
        }
        if dynamic:
            eval_summary["stride"] = stride
        else:
            eval_summary["image_size"] = list(image_size)
            eval_summary["pad"] = pad
            eval_summary["pad_align"] = pad_align
        if args.checkpoint:
            eval_summary["checkpoint"] = str(args.checkpoint)
        else:
            eval_summary["onnx"] = str(args.onnx)
        save_json(output_dir / "eval_results.json", eval_summary)
        print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
