"""独立验证模块 — 加载模型 + 验证集，计算与训练时一致的 val_loss 和 val_iou。

用法示例::

    # PyTorch checkpoint（固定尺寸，和训练验证一致）
    python evaluate.py --checkpoint /path/to/best.pth \
        --images-dir /path/to/images --masks-dir /path/to/masks \
        --val-txt /path/to/val.txt

    # ONNX 模型
    python evaluate.py --onnx /path/to/model.onnx \
        --images-dir /path/to/images --masks-dir /path/to/masks \
        --val-txt /path/to/val.txt \
        --num-classes 1 --imgsz 640 640

    # 动态推理模式（保持原图尺寸，逐张验证）
    python evaluate.py --checkpoint /path/to/best.pth \
        --images-dir /path/to/images --masks-dir /path/to/masks \
        --val-txt /path/to/val.txt --dynamic
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import segmentation_models_pytorch as smp
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
    preprocess_image_array,
    resolve_encoder_weights,
    save_json,
)


# ---------------------------------------------------------------------------
# 逐图 IoU 计算
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_per_image_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    threshold: float,
) -> list[float]:
    """返回 batch 中每张图像各自的 IoU（micro 方式合并类别）。"""
    if num_classes == 1:
        probabilities = torch.sigmoid(logits).squeeze(1)
        predictions = (probabilities > threshold).long()
        target_labels = targets.squeeze(1).long()
        tp, fp, fn, _tn = smp.metrics.get_stats(predictions, target_labels, mode="binary")
    else:
        predictions = torch.argmax(logits, dim=1)
        tp, fp, fn, _tn = smp.metrics.get_stats(
            predictions, targets.long(), mode="multiclass", num_classes=num_classes,
        )
    # tp/fp/fn shape: (batch, num_classes) — sum across classes per image
    tp_sum = tp.sum(dim=1).float()
    fp_sum = fp.sum(dim=1).float()
    fn_sum = fn.sum(dim=1).float()
    denom = tp_sum + fp_sum + fn_sum
    iou_per_image = torch.where(denom > 0, tp_sum / denom, torch.zeros_like(tp_sum))
    return [float(v.item()) for v in iou_per_image]


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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="独立验证模块 — 加载模型 + 验证集，计算 val_loss 和 val_iou",
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
    parser.add_argument("--pad", action="store_true", default=False,
                        help="使用填充模式（默认不开启）")
    parser.add_argument("--pad-align", type=str, default="center",
                        choices=["center", "top_left"],
                        help="填充对齐方式（默认 center）")
    parser.add_argument("--dynamic", action="store_true",
                        help="动态推理：保持原图尺寸，仅填充到 stride 的倍数后推理（逐张处理，无 batch）")
    parser.add_argument("--stride", type=int, default=32,
                        help="动态推理时的对齐步长（默认 32）")

    # 输出
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录，保存 JSON 结果")
    parser.add_argument("--low-iou-thresh", type=float, default=0.5,
                        help="低 IoU 阈值，低于此值的图片将被记录到统计 TXT（默认 0.5）")

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
        image_size = tuple(args.imgsz) if args.imgsz is not None else tuple(checkpoint["image_size"])
        threshold = float(checkpoint.get("threshold", 0.5)) if args.threshold < 0 else args.threshold
        num_classes = int(checkpoint["model_config"]["num_classes"])
        preprocessing = checkpoint["preprocessing"]
        pad = bool(args.pad)
        pad_align = args.pad_align
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
        pad = bool(args.pad)
        pad_align = args.pad_align
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

    # ------------------------------------------------------------------
    # 构建 loss 函数（仅 PyTorch 模式）
    # ------------------------------------------------------------------
    loss_fn = create_loss_function(num_classes) if pytorch_model is not None else None

    low_iou_thresh = float(args.low_iou_thresh)

    total_loss_sum = 0.0
    total_iou_sum = 0.0
    global_tp = 0.0
    global_fp = 0.0
    global_fn = 0.0
    batch_count = 0
    sample_count = 0
    per_image_ious: list[tuple[str, float]] = []   # (image_name, iou)
    started_at = time.time()

    if dynamic:
        # ==============================================================
        # 动态推理模式：逐张处理，保持原图尺寸
        # ==============================================================
        sample_pairs = collect_split_pairs(images_dir, masks_dir, val_txt)
        sample_count = len(sample_pairs)
        print(f"\nValidation dataset: {sample_count} samples (dynamic mode, stride={stride})")
        print(f"Device: {device}, AMP: {use_amp}")

        for idx, (image_path, mask_path) in enumerate(sample_pairs):
            stem = image_path.stem

            # 加载原图 + GT mask
            image = Image.open(image_path).convert("RGB")
            gt_pil = Image.open(mask_path).convert("L")
            orig_w, orig_h = image.size

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

            # 构造原图尺寸的 GT tensor（裁掉填充后计算 loss 和 IoU）
            if num_classes == 1:
                gt_tensor_orig = torch.from_numpy((gt_array > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
            else:
                if mask_values:
                    indexed_orig = np.zeros(gt_array.shape, dtype=np.int64)
                    for ci, mv in enumerate(mask_values):
                        indexed_orig[gt_array == mv] = ci
                    gt_tensor_orig = torch.from_numpy(indexed_orig).unsqueeze(0).long()
                else:
                    gt_tensor_orig = torch.from_numpy(gt_array.astype(np.int64)).unsqueeze(0).long()

            # 推理 + 计算 loss 和 IoU（均在原图尺寸上）
            if pytorch_model is not None:
                input_device = input_tensor.to(device, non_blocking=True)

                with torch.inference_mode():
                    autocast_enabled = use_amp and device.type == "cuda"
                    with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                        logits = pytorch_model(input_device)

                # 裁回原图尺寸计算 loss 和 IoU
                logits_cropped = logits.detach()[:, :, :orig_h, :orig_w]
                gt_orig_device = gt_tensor_orig.to(device, non_blocking=True)
                batch_loss = float(loss_fn(logits_cropped, gt_orig_device).item())
                batch_iou = compute_batch_iou(logits_cropped, gt_orig_device, num_classes, threshold)
                total_loss_sum += batch_loss
                total_iou_sum += batch_iou
                batch_count += 1

                # 累加 Global 统计量（原图尺寸）
                with torch.inference_mode():
                    if num_classes == 1:
                        preds = (torch.sigmoid(logits_cropped).squeeze(1) > threshold).long()
                        targets = gt_orig_device.squeeze(1).long()
                        tp, fp, fn, _ = smp.metrics.get_stats(preds, targets, mode="binary")
                    else:
                        preds = torch.argmax(logits_cropped, dim=1)
                        tp, fp, fn, _ = smp.metrics.get_stats(
                            preds, gt_orig_device.long(), mode="multiclass", num_classes=num_classes,
                        )
                    global_tp += float(tp.sum().item())
                    global_fp += float(fp.sum().item())
                    global_fn += float(fn.sum().item())

                per_image_ious.append((stem, batch_iou))

                print(f"[{idx + 1}/{sample_count}] {stem}  "
                      f"loss={batch_loss:.4f}  iou={batch_iou:.4f}")
            else:
                input_name = onnx_session.get_inputs()[0].name
                outputs = onnx_session.run(None, {input_name: input_tensor.numpy()})
                logits_np = outputs[0]
                logits_tensor = torch.from_numpy(logits_np)

                # 裁回原图尺寸计算 IoU
                logits_cropped = logits_tensor[:, :, :orig_h, :orig_w]
                batch_iou = compute_batch_iou(logits_cropped, gt_tensor_orig, num_classes, threshold)
                total_iou_sum += batch_iou
                batch_count += 1

                # 累加 Global 统计量（原图尺寸）
                if num_classes == 1:
                    preds = (torch.sigmoid(logits_cropped).squeeze(1) > threshold).long()
                    targets = gt_tensor_orig.squeeze(1).long()
                    tp, fp, fn, _ = smp.metrics.get_stats(preds, targets, mode="binary")
                else:
                    preds = torch.argmax(logits_cropped, dim=1)
                    tp, fp, fn, _ = smp.metrics.get_stats(
                        preds, gt_tensor_orig.long(), mode="multiclass", num_classes=num_classes,
                    )
                global_tp += float(tp.sum().item())
                global_fp += float(fp.sum().item())
                global_fn += float(fn.sum().item())

                per_image_ious.append((stem, batch_iou))

                print(f"[{idx + 1}/{sample_count}] {stem}  iou={batch_iou:.4f}")

    else:
        # ==============================================================
        # 固定尺寸模式：使用 DataLoader 批量验证（和训练验证一致）
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

        sample_count = len(val_dataset)
        print(f"\nValidation dataset: {sample_count} samples, {len(val_loader)} batches")
        print(f"Device: {device}, AMP: {use_amp}")

        try:
            enable_live_progress = bool(
                getattr(getattr(__import__("sys"), "stderr", None), "isatty", lambda: False)()
            )
        except Exception:
            enable_live_progress = False

        iterator = tqdm(val_loader, total=len(val_loader), leave=True, disable=not enable_live_progress)

        sample_idx = 0
        for images, masks in iterator:
            cur_batch_size = images.shape[0]
            images_device = images.to(device, non_blocking=True)
            masks_device = masks.to(device, non_blocking=True)

            if pytorch_model is not None:
                with torch.inference_mode():
                    autocast_enabled = use_amp and device.type == "cuda"
                    with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                        logits = pytorch_model(images_device)
                        batch_loss = float(loss_fn(logits, masks_device).item())
                logits_det = logits.detach()
                masks_det = masks_device.detach()
                batch_iou = compute_batch_iou(logits_det, masks_det, num_classes, threshold)
                total_loss_sum += batch_loss
                total_iou_sum += batch_iou
                batch_count += 1
                iterator.set_postfix(loss=f"{batch_loss:.4f}", IoU=f"{batch_iou:.4f}")

                img_ious = compute_per_image_iou(logits_det, masks_det, num_classes, threshold)
                for i, iou_val in enumerate(img_ious):
                    name = val_dataset.samples[sample_idx + i][0].stem
                    per_image_ious.append((name, iou_val))
            else:
                input_name = onnx_session.get_inputs()[0].name
                input_array = images.numpy().astype(np.float32)
                outputs = onnx_session.run(None, {input_name: input_array})
                logits_tensor = torch.from_numpy(outputs[0])
                batch_iou = compute_batch_iou(logits_tensor, masks, num_classes, threshold)
                total_iou_sum += batch_iou
                batch_count += 1
                iterator.set_postfix(IoU=f"{batch_iou:.4f}")

                img_ious = compute_per_image_iou(logits_tensor, masks, num_classes, threshold)
                for i, iou_val in enumerate(img_ious):
                    name = val_dataset.samples[sample_idx + i][0].stem
                    per_image_ious.append((name, iou_val))

            sample_idx += cur_batch_size

    elapsed = time.time() - started_at

    # ------------------------------------------------------------------
    # 输出 val_loss 和 val_iou（与训练验证格式一致）
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Evaluation Summary ({sample_count} images, {elapsed:.1f}s)")
    print(f"{'=' * 60}")

    val_iou = total_iou_sum / batch_count if batch_count > 0 else float("nan")
    global_denom = global_tp + global_fp + global_fn
    val_iou_global = global_tp / global_denom if global_denom > 0 else float("nan")
    if pytorch_model is not None and batch_count > 0:
        val_loss = total_loss_sum / batch_count
        if dynamic:
            print(f"val_loss={val_loss:.4f}")
            print(f"val_iou (Mean)={val_iou:.4f}")
            print(f"val_iou (Global)={val_iou_global:.4f}")
        else:
            print(f"val_loss={val_loss:.4f}  val_iou={val_iou:.4f}")
    elif batch_count > 0:
        val_loss = float("nan")
        if dynamic:
            print(f"val_iou (Mean)={val_iou:.4f}  (ONNX mode, no loss)")
            print(f"val_iou (Global)={val_iou_global:.4f}")
        else:
            print(f"val_iou={val_iou:.4f}  (ONNX mode, no loss)")
    else:
        val_loss = float("nan")
        print("No batches evaluated")

    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # 低 IoU 图片统计
    # ------------------------------------------------------------------
    low_iou_images = [(name, iou) for name, iou in per_image_ious if iou < low_iou_thresh]
    low_iou_images.sort(key=lambda x: x[1])

    print(f"\nLow IoU detection (threshold < {low_iou_thresh:.2f}):")
    print(f"  Total images evaluated: {len(per_image_ious)}")
    print(f"  Images with IoU < {low_iou_thresh:.2f}: {len(low_iou_images)}")
    if per_image_ious:
        all_ious = [iou for _, iou in per_image_ious]
        print(f"  IoU range: [{min(all_ious):.4f}, {max(all_ious):.4f}]")
    if low_iou_images:
        low_ious_only = [iou for _, iou in low_iou_images]
        print(f"  Low IoU mean: {sum(low_ious_only) / len(low_ious_only):.4f}")

    # 生成统计 TXT 文件
    report_dir = output_dir if output_dir is not None else Path(".")
    ensure_dir(report_dir)
    report_path = report_dir / "low_iou_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Low IoU Report (threshold < {low_iou_thresh:.2f})\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Evaluation time: {elapsed:.1f}s\n")
        f.write(f"Total images evaluated: {len(per_image_ious)}\n")
        f.write(f"Images with IoU < {low_iou_thresh:.2f}: {len(low_iou_images)}\n")
        if per_image_ious:
            all_ious = [iou for _, iou in per_image_ious]
            f.write(f"Overall IoU range: [{min(all_ious):.4f}, {max(all_ious):.4f}]\n")
            f.write(f"Overall IoU mean: {sum(all_ious) / len(all_ious):.4f}\n")
        if low_iou_images:
            low_ious_only = [iou for _, iou in low_iou_images]
            f.write(f"Low IoU mean: {sum(low_ious_only) / len(low_ious_only):.4f}\n")
        f.write(f"{'=' * 60}\n\n")
        if low_iou_images:
            f.write(f"{'Image':<60s} {'IoU':>8s}\n")
            f.write(f"{'-' * 60} {'-' * 8}\n")
            for name, iou in low_iou_images:
                f.write(f"{name:<60s} {iou:>8.4f}\n")
        else:
            f.write("No images with IoU below the threshold.\n")
    print(f"\nLow IoU report saved to: {report_path}")

    # 保存 JSON 汇总
    if output_dir is not None:
        eval_summary: dict[str, Any] = {
            "sample_count": sample_count,
            "elapsed_seconds": elapsed,
            "num_classes": num_classes,
            "threshold": threshold,
            "dynamic": dynamic,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "low_iou_thresh": low_iou_thresh,
            "low_iou_count": len(low_iou_images),
        }
        if dynamic:
            eval_summary["val_iou_global"] = val_iou_global
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
