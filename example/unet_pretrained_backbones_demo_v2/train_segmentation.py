import csv
import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import EvalTransform, TrainAugmentation
from common import (
    SegmentationTxtDataset,
    build_model,
    compute_batch_iou,
    create_loss_function,
    ensure_dir,
    get_preprocessing_config,
    load_optional_unet_meta,
    normalize_arch_name,
    resolve_encoder_weights,
    save_checkpoint,
    save_json,
    save_overlay_batch_visualization,
    save_metrics_csv,
    save_train_batch_visualization,
)


SNAPSHOT_EPOCHS = {1, 50, 100}
DEFAULT_AUTO_BATCH_FRACTION = 0.50
AUTO_BATCH_CANDIDATES = [1, 2, 4, 8, 16, 32, 64, 128]


def resolve_stop_signal_path() -> Path | None:
    raw_path = str(os.getenv("XANY_SEMSEG_STOP_SIGNAL_FILE", "") or "").strip()
    if not raw_path:
        return None
    try:
        return Path(raw_path)
    except Exception:
        return None


def is_stop_requested(stop_signal_path: Path | None) -> bool:
    if stop_signal_path is None:
        return False
    try:
        return stop_signal_path.exists()
    except Exception:
        return False


def resolve_auto_batch_fraction(requested_fraction: float | None = None) -> float:
    if requested_fraction is not None:
        try:
            normalized_fraction = float(requested_fraction)
            if 0.1 <= normalized_fraction <= 0.95:
                return normalized_fraction
        except Exception:
            pass
    fraction = DEFAULT_AUTO_BATCH_FRACTION
    try:
        env_fraction = float(os.getenv("XANY_SEMSEG_AUTOBATCH_FRACTION", str(DEFAULT_AUTO_BATCH_FRACTION)))
        if 0.1 <= env_fraction <= 0.95:
            fraction = env_fraction
    except Exception:
        pass
    return float(fraction)


def release_cuda_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def get_cuda_memory_snapshot(device: torch.device) -> tuple[int, int, int, int]:
    properties = torch.cuda.get_device_properties(device)
    total_bytes = int(getattr(properties, "total_memory", 0) or 0)
    reserved_bytes = int(torch.cuda.memory_reserved(device))
    allocated_bytes = int(torch.cuda.memory_allocated(device))
    free_bytes = 0
    mem_get_info = getattr(torch.cuda, "mem_get_info", None)
    if callable(mem_get_info):
        try:
            free_bytes, total_from_driver = mem_get_info(device)
            total_bytes = int(total_from_driver or total_bytes)
        except Exception:
            free_bytes = 0
    if free_bytes <= 0 and total_bytes > 0:
        free_bytes = max(total_bytes - max(reserved_bytes, allocated_bytes), 0)
    return total_bytes, free_bytes, reserved_bytes, allocated_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, default=r"/hy-tmp/code4/3")
    parser.add_argument("--masks-dir", type=str, default=r"/hy-tmp/code4/masks")
    parser.add_argument("--train-txt", type=str, default=r"/hy-tmp/code4/train.txt")
    parser.add_argument("--val-txt", type=str, default=r"/hy-tmp/code4/val.txt")
    parser.add_argument("--save-dir", type=str, default=r"/hy-tmp/code4/runs")
    parser.add_argument("--project-name", type=str, default="")
    parser.add_argument("--arch", type=str, default="Unet")
    parser.add_argument("--encoder-name", type=str, default="resnet18")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--encoder-depth", type=int, default=5)
    parser.add_argument("--encoder-output-stride", type=int, default=16)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--force-single-class", action="store_true")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--pad", action="store_true",
                        help="不缩放，直接把原图放到模型尺寸画布上、不足处用黑色填充")
    parser.add_argument("--pad-align", type=str, default="center",
                        choices=["center", "top_left"],
                        help="填充时原图的对齐方式：center 居中，top_left 放在左上角")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--auto-batch-fraction", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-visual-items", type=int, default=16)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--vflip-prob", type=float, default=0.5)
    parser.add_argument("--rotate90-prob", type=float, default=0.5)
    parser.add_argument("--shift-prob", type=float, default=0.5)
    parser.add_argument("--max-shift-ratio", type=float, default=0.1)
    parser.add_argument("--scale-prob", type=float, default=0.5)
    parser.add_argument("--min-scale", type=float, default=0.9)
    parser.add_argument("--max-scale", type=float, default=1.1)
    parser.add_argument("--noise-prob", type=float, default=0.3)
    parser.add_argument("--noise-std", type=float, default=8.0)
    parser.add_argument("--brightness-prob", type=float, default=0.3)
    parser.add_argument("--brightness-min", type=float, default=0.85)
    parser.add_argument("--brightness-max", type=float, default=1.15)
    parser.add_argument("--contrast-prob", type=float, default=0.3)
    parser.add_argument("--contrast-min", type=float, default=0.85)
    parser.add_argument("--contrast-max", type=float, default=1.15)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    raw_name = str(device_name or "").strip()
    name = raw_name.lower()
    if not name or name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name in ("cpu", "-1", "none"):
        return torch.device("cpu")
    if name in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "," in name:
        name = name.split(",", 1)[0].strip()
    if name.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{int(name)}")
        return torch.device("cpu")
    if name.startswith("cuda") and (not torch.cuda.is_available()):
        return torch.device("cpu")
    try:
        return torch.device(name)
    except Exception as exc:
        raise RuntimeError(f"Invalid device string: {raw_name!r}") from exc


def resolve_cuda_device_ids(device_name: str) -> list[int]:
    if not torch.cuda.is_available():
        return []
    max_devices = int(torch.cuda.device_count() or 0)
    if max_devices <= 0:
        return []

    name = str(device_name or "").strip().lower()
    if not name or name in ("auto", "cuda", "gpu"):
        return [0]
    if name in ("cpu", "-1", "none"):
        return []

    raw_parts: list[str] = []
    if "," in name:
        raw_parts = [part.strip() for part in name.split(",") if str(part or "").strip()]
    elif name.isdigit():
        raw_parts = [name]
    elif name.startswith("cuda:"):
        suffix = name.split(":", 1)[1].strip()
        if suffix.isdigit():
            raw_parts = [suffix]

    device_ids: list[int] = []
    for part in raw_parts:
        token = str(part or "").strip().lower()
        if token.startswith("cuda:"):
            token = token.split(":", 1)[1].strip()
        if not token.isdigit():
            continue
        idx = int(token)
        if 0 <= idx < max_devices and idx not in device_ids:
            device_ids.append(idx)

    return device_ids


def is_cuda_out_of_memory_error(error: BaseException) -> bool:
    message = str(error or "").lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def build_grad_scaler(device: torch.device, enabled: bool) -> object | None:
    if device.type != "cuda":
        return None
    amp_module = getattr(torch, "amp", None)
    grad_scaler_cls = getattr(amp_module, "GradScaler", None) if amp_module is not None else None
    if grad_scaler_cls is not None:
        try:
            return grad_scaler_cls("cuda", enabled=enabled)
        except TypeError:
            try:
                return grad_scaler_cls(device="cuda", enabled=enabled)
            except TypeError:
                pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def resolve_decoder_channels(arch: str, encoder_depth: int) -> tuple[int, ...] | None:
    if normalize_arch_name(arch) in {"Unet", "UnetPlusPlus"}:
        safe_depth = max(1, int(encoder_depth))
        return (256, 128, 64, 32, 16)[:safe_depth]
    return None


def build_training_model(
    args: argparse.Namespace,
    effective_num_classes: int,
    decoder_channels: tuple[int, ...] | None,
    device: torch.device,
    device_ids: list[int],
) -> torch.nn.Module:
    model = build_model(
        arch=args.arch,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=args.in_channels,
        num_classes=effective_num_classes,
        encoder_depth=args.encoder_depth,
        encoder_output_stride=args.encoder_output_stride,
        decoder_channels=decoder_channels,
    )
    model = model.to(device)
    if device.type == "cuda" and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    return model


def resolve_auto_batch_size(
    model: torch.nn.Module,
    loss_fn,
    device: torch.device,
    in_channels: int,
    image_size: tuple[int, int],
    num_classes: int,
    use_amp: bool,
    learning_rate: float,
    weight_decay: float,
    target_fraction: float,
) -> int:
    if device.type != "cuda":
        fallback_batch_size = 1
        print(
            f"Auto batch requested but device={device} is not CUDA; using batch_size={fallback_batch_size}.",
            flush=True,
        )
        return fallback_batch_size

    print(
        f"Auto batch requested, targeting about {target_fraction * 100:.0f}% of currently available CUDA memory...",
        flush=True,
    )
    original_mode = bool(getattr(model, "training", True))
    resolved_batch_size = 0
    image_height, image_width = int(image_size[0]), int(image_size[1])
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = build_grad_scaler(device, enabled=use_amp)
    release_cuda_memory()
    total_bytes, free_bytes, reserved_bytes, allocated_bytes = get_cuda_memory_snapshot(device)
    target_process_bytes = 0
    if total_bytes > 0 and free_bytes > 0:
        target_process_bytes = max(reserved_bytes, allocated_bytes) + int(free_bytes * target_fraction)
        print(
            "Auto batch CUDA memory snapshot: "
            f"total={total_bytes / (1 << 30):.2f}G, "
            f"free={free_bytes / (1 << 30):.2f}G, "
            f"reserved={reserved_bytes / (1 << 30):.2f}G, "
            f"allocated={allocated_bytes / (1 << 30):.2f}G, "
            f"target={target_process_bytes / (1 << 30):.2f}G",
            flush=True,
        )

    try:
        for batch_size in AUTO_BATCH_CANDIDATES:
            images = None
            masks = None
            logits = None
            loss = None
            try:
                optimizer.zero_grad(set_to_none=True)
                try:
                    model.zero_grad(set_to_none=True)
                except Exception:
                    pass
                release_cuda_memory()
                reset_peak_memory_stats = getattr(torch.cuda, "reset_peak_memory_stats", None)
                if callable(reset_peak_memory_stats):
                    try:
                        reset_peak_memory_stats(device)
                    except Exception:
                        pass

                images = torch.randn(batch_size, in_channels, image_height, image_width, device=device)
                if num_classes == 1:
                    masks = torch.zeros(batch_size, 1, image_height, image_width, device=device, dtype=torch.float32)
                else:
                    masks = torch.zeros(batch_size, image_height, image_width, device=device, dtype=torch.long)

                model.train()
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                    loss = loss_fn(logits, masks)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    pass
                peak_reserved_reader = getattr(torch.cuda, "max_memory_reserved", None)
                peak_allocated_reader = getattr(torch.cuda, "max_memory_allocated", None)
                peak_process_bytes = max(
                    int(peak_reserved_reader(device)) if callable(peak_reserved_reader) else 0,
                    int(peak_allocated_reader(device)) if callable(peak_allocated_reader) else 0,
                    int(torch.cuda.memory_reserved(device)),
                    int(torch.cuda.memory_allocated(device)),
                )
                if target_process_bytes > 0:
                    print(
                        f"Auto batch probe passed: batch_size={batch_size}, "
                        f"peak={peak_process_bytes / (1 << 30):.2f}G / target={target_process_bytes / (1 << 30):.2f}G",
                        flush=True,
                    )
                    if peak_process_bytes > target_process_bytes:
                        if resolved_batch_size < 1:
                            resolved_batch_size = batch_size
                        print(
                            f"Auto batch target reached; keeping batch_size={resolved_batch_size}.",
                            flush=True,
                        )
                        break
                else:
                    print(f"Auto batch probe passed: batch_size={batch_size}", flush=True)
                resolved_batch_size = batch_size
            except RuntimeError as exc:
                if is_cuda_out_of_memory_error(exc):
                    print(f"Auto batch probe OOM at batch_size={batch_size}", flush=True)
                    break
                raise
            finally:
                try:
                    optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass
                try:
                    model.zero_grad(set_to_none=True)
                except Exception:
                    pass
                del images, masks, logits, loss
                release_cuda_memory()

        if resolved_batch_size < 1:
            raise RuntimeError(
                "Auto batch failed because the model ran out of memory even at batch_size=1. "
                "Reduce image size or choose a different device."
            )

        print(
            f"Auto batch selected: batch_size={resolved_batch_size} at ~{target_fraction * 100:.0f}% CUDA memory target.",
            flush=True,
        )
        return resolved_batch_size
    finally:
        try:
            optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
        try:
            model.zero_grad(set_to_none=True)
        except Exception:
            pass
        try:
            model.train(original_mode)
        except Exception:
            pass
        del optimizer, scaler
        release_cuda_memory()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(
    images_dir: Path,
    masks_dir: Path,
    split_txt: Path,
    image_size: tuple[int, int],
    num_classes: int,
    preprocessing: dict,
    mask_values: list[int] | None,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
    transform,
    pad: bool = False,
    pad_align: str = "center",
) -> DataLoader:
    dataset = SegmentationTxtDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        split_txt=split_txt,
        image_size=image_size,
        num_classes=num_classes,
        preprocessing=preprocessing,
        mask_values=mask_values,
        transform=transform,
        pad=pad,
        pad_align=pad_align,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def save_train_batches(
    train_loader: DataLoader,
    preprocessing: dict,
    run_dir: Path,
    max_visual_items: int,
) -> None:
    for batch_index, (images, masks) in enumerate(train_loader):
        save_train_batch_visualization(
            images=images,
            masks=masks,
            preprocessing=preprocessing,
            output_path=run_dir / f"train_batch{batch_index}.png",
            max_items=max_visual_items,
        )
        if batch_index >= 1:
            break


def save_validation_snapshot(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    preprocessing: dict[str, Any],
    epoch: int,
    run_dir: Path,
    max_visual_items: int,
    save_label: bool = True,
    save_prediction: bool = True,
) -> None:
    if save_label:
        save_overlay_batch_visualization(
            images=images,
            masks=labels,
            preprocessing=preprocessing,
            output_path=run_dir / f"val_batch_label_ep{epoch}.png",
            max_items=max_visual_items,
            style="prediction",
            panels=("image", "overlay", "mask"),
            show_text=False,
        )
    if save_prediction:
        save_overlay_batch_visualization(
            images=images,
            masks=predictions,
            preprocessing=preprocessing,
            output_path=run_dir / f"val_batch_pred_ep{epoch}.png",
            max_items=max_visual_items,
            style="prediction",
            panels=("image", "overlay", "mask"),
            show_text=False,
        )


def logits_to_batch_masks(
    logits: torch.Tensor,
    num_classes: int,
    threshold: float,
) -> torch.Tensor:
    if num_classes == 1:
        probabilities = torch.sigmoid(logits)
        return (probabilities > threshold).float()
    return torch.argmax(logits, dim=1, keepdim=True).float()


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn,
    device: torch.device,
    num_classes: int,
    threshold: float,
    optimizer: torch.optim.Optimizer | None,
    preprocessing: dict,
    use_amp: bool = False,
    scaler: object | None = None,
    snapshot_epoch: int | None = None,
    snapshot_dir: Path | None = None,
    max_visual_items: int = 4,
    snapshot_save_label: bool = True,
    snapshot_save_prediction: bool = True,
) -> tuple[float, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_iou = 0.0
    snapshot_saved = False

    try:
        enable_live_progress = bool(
            getattr(getattr(sys, "stderr", None), "isatty", lambda: False)()
        )
    except Exception:
        enable_live_progress = False

    iterator = tqdm(
        dataloader,
        total=len(dataloader),
        leave=False,
        disable=not enable_live_progress,
    )
    for images, masks in iterator:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        autocast_enabled = use_amp and device.type == "cuda"
        predictions = None

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(images)
                loss = loss_fn(logits, masks)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.inference_mode():
                with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                    logits = model(images)
                    loss = loss_fn(logits, masks)
                if snapshot_epoch is not None and snapshot_dir is not None and not snapshot_saved:
                    predictions = logits_to_batch_masks(logits, num_classes, threshold).detach().cpu()
                    save_validation_snapshot(
                        images=images,
                        labels=masks,
                        predictions=predictions,
                        preprocessing=preprocessing,
                        epoch=snapshot_epoch,
                        run_dir=snapshot_dir,
                        max_visual_items=max_visual_items,
                        save_label=bool(snapshot_save_label),
                        save_prediction=bool(snapshot_save_prediction),
                    )
                    snapshot_saved = True

        total_loss += float(loss.item())
        total_iou += compute_batch_iou(logits.detach(), masks.detach(), num_classes, threshold)
        iterator.set_postfix(loss=f"{loss.item():.4f}")
        del images, masks, logits, loss, predictions

    return total_loss / len(dataloader), total_iou / len(dataloader)


def build_project_name(args: argparse.Namespace) -> str:
    if args.project_name:
        return args.project_name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    arch_name = normalize_arch_name(args.arch).lower()
    return f"{timestamp}_{arch_name}_{args.encoder_name}"


def load_metric_rows_from_csv(csv_path: str | Path) -> list[dict[str, float]]:
    source = Path(csv_path)
    if not source.is_file():
        return []
    rows: list[dict[str, float]] = []
    with source.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            if not isinstance(raw_row, dict):
                continue
            try:
                epoch = int(float(raw_row.get("epoch") or 0))
            except Exception:
                continue
            parsed_row: dict[str, float] = {"epoch": float(epoch)}
            for key in ("train_loss", "val_loss", "train_iou", "val_iou", "lr"):
                try:
                    parsed_row[key] = float(raw_row.get(key) or 0.0)
                except Exception:
                    parsed_row[key] = 0.0
            rows.append(parsed_row)
    return rows


def style_metric_axis(axis, title: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#0f172a")
    axis.set_xlabel("Epoch", color="#475569")
    axis.set_ylabel(ylabel, color="#475569")
    axis.grid(True, color="#e2e8f0", linewidth=0.8, alpha=0.9)
    axis.set_facecolor("#ffffff")
    for spine in axis.spines.values():
        spine.set_color("#cbd5e1")
    axis.tick_params(colors="#334155")


def plot_metric_pair(
    axis,
    epochs: np.ndarray,
    train_values: np.ndarray,
    val_values: np.ndarray,
    train_label: str,
    val_label: str,
    train_color: str,
    val_color: str,
) -> None:
    axis.plot(epochs, train_values, color=train_color, linewidth=2.2, marker="o", markersize=3.8, label=train_label)
    axis.plot(epochs, val_values, color=val_color, linewidth=2.2, marker="o", markersize=3.8, label=val_label)
    axis.legend(frameon=False, loc="best")


def save_results_plot(results_csv_path: str | Path, output_path: str | Path) -> None:
    metric_rows = load_metric_rows_from_csv(results_csv_path)
    if not metric_rows:
        return
    epochs = np.asarray([int(float(row.get("epoch") or 0.0)) for row in metric_rows], dtype=np.int32)
    train_loss = np.asarray([float(row.get("train_loss") or 0.0) for row in metric_rows], dtype=np.float32)
    val_loss = np.asarray([float(row.get("val_loss") or 0.0) for row in metric_rows], dtype=np.float32)
    train_iou = np.asarray([float(row.get("train_iou") or 0.0) for row in metric_rows], dtype=np.float32)
    val_iou = np.asarray([float(row.get("val_iou") or 0.0) for row in metric_rows], dtype=np.float32)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    figure, axes = plt.subplots(1, 2, figsize=(13.6, 6.0), dpi=220, facecolor="#f8fafc")
    figure.patch.set_facecolor("#f8fafc")
    style_metric_axis(axes[0], title="Loss Curves", ylabel="Loss")
    style_metric_axis(axes[1], title="IoU Curves", ylabel="IoU")
    plot_metric_pair(
        axes[0],
        epochs,
        train_loss,
        val_loss,
        train_label="Train Loss",
        val_label="Val Loss",
        train_color="#2563eb",
        val_color="#f97316",
    )
    plot_metric_pair(
        axes[1],
        epochs,
        train_iou,
        val_iou,
        train_label="Train IoU",
        val_label="Val IoU",
        train_color="#0891b2",
        val_color="#db2777",
    )

    best_index = int(np.argmax(val_iou)) if val_iou.size else 0
    axes[1].scatter([epochs[best_index]], [val_iou[best_index]], s=64, color="#be185d", edgecolors="#ffffff", linewidths=1.2, zorder=5)
    figure.tight_layout(rect=(0.02, 0.04, 0.98, 0.98))
    destination = Path(output_path)
    ensure_dir(destination.parent)
    figure.savefig(destination, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)


def emit_metrics_line(epoch_metrics: dict[str, float]) -> None:
    payload = {
        "epoch": int(epoch_metrics.get("epoch") or 0),
        "metrics": {
            "train/loss": float(epoch_metrics.get("train_loss") or 0.0),
            "val/loss": float(epoch_metrics.get("val_loss") or 0.0),
            "train/iou": float(epoch_metrics.get("train_iou") or 0.0),
            "val/iou": float(epoch_metrics.get("val_iou") or 0.0),
        },
    }
    print("METRICS " + json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    args.encoder_weights = resolve_encoder_weights(args.encoder_name, args.encoder_weights)
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    device_ids = resolve_cuda_device_ids(args.device)
    effective_num_classes = 1 if args.force_single_class else args.num_classes
    use_amp = args.amp and device.type == "cuda"
    scaler = build_grad_scaler(device, enabled=use_amp)
    image_size = (args.height, args.width)
    save_root = ensure_dir(args.save_dir)
    run_dir = ensure_dir(save_root / build_project_name(args))
    weight_dir = ensure_dir(run_dir / "weight")
    stop_signal_path = resolve_stop_signal_path()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    train_txt = Path(args.train_txt)
    val_txt = Path(args.val_txt)
    unet_meta = load_optional_unet_meta(train_txt) or load_optional_unet_meta(val_txt)
    classes = [
        str(item)
        for item in list((unet_meta.get("classes") if isinstance(unet_meta, dict) else []) or [])
        if str(item or "").strip()
    ]
    single_cls = bool(unet_meta.get("single_cls")) if isinstance(unet_meta, dict) else bool(args.force_single_class)
    try:
        meta_num_classes = int(unet_meta.get("num_classes") or 0) if isinstance(unet_meta, dict) else 0
    except Exception:
        meta_num_classes = 0
    if meta_num_classes <= 0:
        meta_num_classes = int(effective_num_classes)
    raw_mask_values = unet_meta.get("mask_values") if isinstance(unet_meta, dict) else None
    mask_values = [int(x) for x in list(raw_mask_values or [])] if raw_mask_values else []

    preprocessing = get_preprocessing_config(args.encoder_name, args.encoder_weights)
    pin_memory = device.type == "cuda"
    requested_batch_size = int(args.batch_size)
    decoder_channels = resolve_decoder_channels(args.arch, args.encoder_depth)
    auto_batch_fraction = resolve_auto_batch_fraction(args.auto_batch_fraction)

    if requested_batch_size < 1:
        probe_model = build_training_model(args, effective_num_classes, decoder_channels, device, device_ids)
        probe_loss_fn = create_loss_function(effective_num_classes)
        try:
            args.batch_size = resolve_auto_batch_size(
                model=probe_model,
                loss_fn=probe_loss_fn,
                device=device,
                in_channels=args.in_channels,
                image_size=image_size,
                num_classes=effective_num_classes,
                use_amp=use_amp,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                target_fraction=auto_batch_fraction,
            )
        finally:
            del probe_loss_fn, probe_model
            release_cuda_memory()
    else:
        args.batch_size = requested_batch_size

    train_transform = TrainAugmentation(
        image_size=image_size,
        hflip_prob=args.hflip_prob,
        vflip_prob=args.vflip_prob,
        rotate90_prob=args.rotate90_prob,
        shift_prob=args.shift_prob,
        max_shift_ratio=args.max_shift_ratio,
        scale_prob=args.scale_prob,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        noise_prob=args.noise_prob,
        noise_std=args.noise_std,
        brightness_prob=args.brightness_prob,
        brightness_range=(args.brightness_min, args.brightness_max),
        contrast_prob=args.contrast_prob,
        contrast_range=(args.contrast_min, args.contrast_max),
        pad=args.pad,
        pad_align=args.pad_align,
    )
    val_transform = EvalTransform(image_size=image_size, pad=args.pad, pad_align=args.pad_align)

    train_loader = build_dataloader(
        images_dir,
        masks_dir,
        train_txt,
        image_size,
        effective_num_classes,
        preprocessing,
        mask_values,
        args.batch_size,
        args.num_workers,
        True,
        pin_memory,
        train_transform,
        pad=args.pad,
        pad_align=args.pad_align,
    )
    val_loader = build_dataloader(
        images_dir,
        masks_dir,
        val_txt,
        image_size,
        effective_num_classes,
        preprocessing,
        mask_values,
        args.batch_size,
        args.num_workers,
        False,
        pin_memory,
        val_transform,
        pad=args.pad,
        pad_align=args.pad_align,
    )

    save_train_batches(train_loader, preprocessing, run_dir, args.max_visual_items)

    model = build_training_model(args, effective_num_classes, decoder_channels, device, device_ids)

    loss_fn = create_loss_function(effective_num_classes)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    model_config = {
        "arch": normalize_arch_name(args.arch),
        "encoder_name": args.encoder_name,
        "encoder_weights": args.encoder_weights,
        "encoder_depth": args.encoder_depth,
        "encoder_output_stride": args.encoder_output_stride,
        "decoder_channels": list(decoder_channels or []),
        "in_channels": args.in_channels,
        "num_classes": effective_num_classes,
        "requested_num_classes": args.num_classes,
        "force_single_class": args.force_single_class,
    }

    config_payload = {
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "train_txt": str(train_txt),
        "val_txt": str(val_txt),
        "model_config": model_config,
        "image_size": list(image_size),
        "pad": bool(args.pad),
        "pad_align": str(args.pad_align),
        "batch_size": args.batch_size,
        "requested_batch_size": requested_batch_size,
        "auto_batch_fraction": auto_batch_fraction,
        "stop_signal_file": str(stop_signal_path) if stop_signal_path is not None else "",
        "epochs": args.epochs,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "threshold": args.threshold,
        "amp": use_amp,
        "amp_requested": args.amp,
        "requested_num_classes": args.num_classes,
        "effective_num_classes": effective_num_classes,
        "force_single_class": args.force_single_class,
        "device": str(device),
        "requested_device": str(args.device or ""),
        "device_ids": list(device_ids),
        "preprocessing": preprocessing,
        "classes": classes,
        "mask_values": mask_values,
        "single_cls": bool(single_cls),
        "augmentations": {
            "hflip_prob": args.hflip_prob,
            "vflip_prob": args.vflip_prob,
            "rotate90_prob": args.rotate90_prob,
            "shift_prob": args.shift_prob,
            "max_shift_ratio": args.max_shift_ratio,
            "scale_prob": args.scale_prob,
            "min_scale": args.min_scale,
            "max_scale": args.max_scale,
            "noise_prob": args.noise_prob,
            "noise_std": args.noise_std,
            "brightness_prob": args.brightness_prob,
            "brightness_min": args.brightness_min,
            "brightness_max": args.brightness_max,
            "contrast_prob": args.contrast_prob,
            "contrast_min": args.contrast_min,
            "contrast_max": args.contrast_max,
        },
    }
    save_json(run_dir / "config.json", config_payload)
    if isinstance(unet_meta, dict) and unet_meta:
        save_json(run_dir / "unet_meta.json", unet_meta)

    results_csv_path = run_dir / "results.csv"
    metric_results_csv_path = run_dir / "metric_results.csv"

    metric_rows: list[dict[str, float]] = []
    best_val_iou = -1.0
    best_epoch = 0
    best_checkpoint_path = weight_dir / "best.pth"
    last_checkpoint_path = weight_dir / "last.pth"
    started_at = time.time()
    stop_requested = False
    stop_epoch = 0

    for epoch in range(1, args.epochs + 1):
        if is_stop_requested(stop_signal_path):
            stop_requested = True
            print(
                f"Graceful stop requested before epoch {epoch}; stopping without starting a new epoch.",
                flush=True,
            )
            break

        train_loss, train_iou = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            effective_num_classes,
            args.threshold,
            optimizer,
            preprocessing,
            use_amp=use_amp,
            scaler=scaler,
        )

        save_label_snapshot = epoch in SNAPSHOT_EPOCHS
        save_prediction_snapshot = bool(save_label_snapshot or epoch == args.epochs or is_stop_requested(stop_signal_path))
        snapshot_epoch = epoch if (save_label_snapshot or save_prediction_snapshot) else None
        val_loss, val_iou = run_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            effective_num_classes,
            args.threshold,
            optimizer=None,
            preprocessing=preprocessing,
            use_amp=use_amp,
            snapshot_epoch=snapshot_epoch,
            snapshot_dir=run_dir,
            max_visual_items=args.max_visual_items,
            snapshot_save_label=save_label_snapshot,
            snapshot_save_prediction=save_prediction_snapshot,
        )
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_iou": train_iou,
            "val_iou": val_iou,
            "lr": optimizer.param_groups[0]["lr"],
        }
        metric_rows.append(epoch_metrics)
        save_metrics_csv(results_csv_path, metric_rows)
        save_results_plot(results_csv_path, run_dir / "results.png")
        emit_metrics_line(epoch_metrics)
        print(epoch_metrics)

        save_checkpoint(
            checkpoint_path=last_checkpoint_path,
            model=model,
            model_config=model_config,
            image_size=image_size,
            threshold=args.threshold,
            preprocessing=preprocessing,
            metrics=epoch_metrics,
            mask_values=mask_values,
            classes=classes,
            single_cls=single_cls,
            num_classes=meta_num_classes,
            pad=args.pad,
            pad_align=args.pad_align,
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                model_config=model_config,
                image_size=image_size,
                threshold=args.threshold,
                preprocessing=preprocessing,
                metrics=epoch_metrics,
                mask_values=mask_values,
                classes=classes,
                single_cls=single_cls,
                num_classes=meta_num_classes,
                pad=args.pad,
                pad_align=args.pad_align,
            )

        if is_stop_requested(stop_signal_path):
            stop_requested = True
            stop_epoch = epoch
            print(
                f"Graceful stop requested; current epoch {epoch} finished, checkpoints saved, stopping now.",
                flush=True,
            )
            break

    metric_payload = {
        "epochs": metric_rows,
        "summary": {
            "best_epoch": best_epoch,
            "best_val_iou": best_val_iou,
            "elapsed_seconds": time.time() - started_at,
            "best_checkpoint": str(best_checkpoint_path),
            "last_checkpoint": str(last_checkpoint_path),
            "stopped_early": stop_requested,
            "stop_epoch": stop_epoch,
        },
    }
    save_json(run_dir / "metric_results.json", metric_payload)
    save_metrics_csv(results_csv_path, metric_rows)
    save_metrics_csv(metric_results_csv_path, metric_rows)
    save_results_plot(results_csv_path, run_dir / "results.png")
    print(metric_payload["summary"])


if __name__ == "__main__":
    main()
