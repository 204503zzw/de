import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_DECODER_CHANNELS = (256, 128, 64, 32, 16)
ARCH_ALIASES = {
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "unet++": "UnetPlusPlus",
    "segformer": "Segformer",
    "deeplabv3": "DeepLabV3",
}
VISUAL_THEMES: dict[str, dict[str, Any]] = {
    "train": {
        "accent": (37, 99, 235),
        "palette": [
            (56, 189, 248),
            (16, 185, 129),
            (250, 204, 21),
            (168, 85, 247),
            (244, 114, 182),
            (249, 115, 22),
        ],
        "card_background": (255, 255, 255),
        "canvas_background": (240, 244, 248),
        "preview_background": (15, 23, 42),
        "text": (15, 23, 42),
        "subtle_text": (71, 85, 105),
        "border": (203, 213, 225),
    },
    "label": {
        "accent": (14, 165, 233),
        "palette": [
            (59, 130, 246),
            (16, 185, 129),
            (250, 204, 21),
            (139, 92, 246),
            (244, 114, 182),
            (249, 115, 22),
        ],
        "card_background": (255, 255, 255),
        "canvas_background": (240, 244, 248),
        "preview_background": (15, 23, 42),
        "text": (15, 23, 42),
        "subtle_text": (71, 85, 105),
        "border": (203, 213, 225),
    },
    "prediction": {
        "accent": (236, 72, 153),
        "palette": [
            (236, 72, 153),
            (59, 130, 246),
            (245, 158, 11),
            (20, 184, 166),
            (139, 92, 246),
            (244, 63, 94),
        ],
        "card_background": (255, 255, 255),
        "canvas_background": (248, 242, 246),
        "preview_background": (30, 27, 75),
        "text": (15, 23, 42),
        "subtle_text": (71, 85, 105),
        "border": (203, 213, 225),
    },
}


def normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized.lower() in {"", "none", "null"}:
        return None
    return normalized


def resolve_encoder_weights(
    encoder_name: str,
    encoder_weights: Any,
) -> str | bool | None:
    normalized_encoder_name = str(encoder_name or "").strip().lower()
    if isinstance(encoder_weights, bool):
        if normalized_encoder_name.startswith("tu-"):
            return True if encoder_weights else None
        return "imagenet" if encoder_weights else None
    normalized_weights = normalize_optional_string(encoder_weights)
    if normalized_encoder_name.startswith("tu-"):
        return None if normalized_weights is None else True
    return normalized_weights


def normalize_arch_name(arch: str) -> str:
    key = arch.strip().lower().replace(" ", "")
    return ARCH_ALIASES.get(key, arch)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    fieldnames = ["epoch", "train_loss", "val_loss", "train_iou", "val_iou", "lr"]
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def scan_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def build_file_maps(directory: str | Path) -> tuple[dict[str, Path], dict[str, Path]]:
    files = scan_files(directory)
    by_name = {path.name: path for path in files}
    by_stem = {path.stem: path for path in files}
    return by_name, by_stem


def read_split_tokens(split_txt: str | Path) -> list[str]:
    with Path(split_txt).open("r", encoding="utf-8") as file:
        tokens = [line.strip() for line in file if line.strip()]
    if not tokens:
        raise ValueError(f"No sample ids found in {split_txt}")
    return tokens


def resolve_sample_path(
    token: str,
    root_dir: str | Path,
    by_name: dict[str, Path],
    by_stem: dict[str, Path],
) -> Path:
    direct_path = Path(root_dir) / token
    if direct_path.is_file():
        return direct_path
    basename = Path(token).name
    stem = Path(token).stem
    if basename in by_name:
        return by_name[basename]
    if stem in by_stem:
        return by_stem[stem]
    raise FileNotFoundError(f"Could not resolve sample '{token}'")


def load_optional_mask_index(split_txt: str | Path) -> dict[str, Path]:
    mask_index_path = Path(split_txt).with_name("mask_index.json")
    if not mask_index_path.is_file():
        return {}
    try:
        raw_index = load_json(mask_index_path)
    except Exception:
        return {}
    resolved_index: dict[str, Path] = {}
    if not isinstance(raw_index, dict):
        return resolved_index
    for image_path, mask_path in raw_index.items():
        try:
            image_key = str(Path(str(image_path)).resolve())
        except Exception:
            image_key = str(image_path or "").strip()
        if not image_key:
            continue
        candidate = Path(str(mask_path or "").strip())
        if not candidate.is_absolute():
            candidate = mask_index_path.parent / candidate
        if not candidate.is_file():
            continue
        try:
            resolved_index[image_key] = candidate.resolve()
        except Exception:
            resolved_index[image_key] = candidate
    return resolved_index


def load_optional_unet_meta(split_txt: str | Path) -> dict[str, Any]:
    meta_path = Path(split_txt).with_name("unet_meta.json")
    if not meta_path.is_file():
        return {}
    try:
        payload = load_json(meta_path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve_mask_path(
    token: str,
    image_path: Path,
    masks_dir: str | Path,
    by_name: dict[str, Path],
    by_stem: dict[str, Path],
    mask_index: dict[str, Path],
) -> Path:
    candidate_keys: list[str] = []
    for value in (token, str(image_path)):
        try:
            candidate_keys.append(str(Path(str(value)).resolve()))
        except Exception:
            normalized = str(value or "").strip()
            if normalized:
                candidate_keys.append(normalized)
    for key in candidate_keys:
        resolved = mask_index.get(key)
        if resolved is not None and Path(resolved).is_file():
            return Path(resolved)
    return resolve_sample_path(token, masks_dir, by_name, by_stem)


def collect_split_pairs(
    images_dir: str | Path,
    masks_dir: str | Path,
    split_txt: str | Path,
) -> list[tuple[Path, Path]]:
    image_by_name, image_by_stem = build_file_maps(images_dir)
    mask_by_name, mask_by_stem = build_file_maps(masks_dir)
    mask_index = load_optional_mask_index(split_txt)
    tokens = read_split_tokens(split_txt)
    pairs = []
    for token in tokens:
        image_path = resolve_sample_path(token, images_dir, image_by_name, image_by_stem)
        mask_path = resolve_mask_path(token, image_path, masks_dir, mask_by_name, mask_by_stem, mask_index)
        pairs.append((image_path, mask_path))
    return pairs


def get_preprocessing_config(
    encoder_name: str,
    encoder_weights: str | None,
) -> dict[str, Any]:
    resolved_weights = resolve_encoder_weights(encoder_name, encoder_weights)
    if resolved_weights is None:
        return {
            "input_space": "RGB",
            "input_range": [0, 1],
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        }
    return smp.encoders.get_preprocessing_params(encoder_name, pretrained=resolved_weights)


def preprocess_image_array(image: np.ndarray, preprocessing: dict[str, Any]) -> np.ndarray:
    processed = image.astype(np.float32)
    input_space = preprocessing.get("input_space", "RGB")
    input_range = preprocessing.get("input_range", [0, 1])
    mean = np.array(preprocessing.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
    std = np.array(preprocessing.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)

    if input_space == "BGR":
        processed = processed[..., ::-1].copy()

    if input_range is not None and processed.max() > 1.0 and input_range[1] == 1:
        processed = processed / 255.0

    processed = processed - mean
    processed = processed / std
    return processed.astype(np.float32)


def denormalize_image_tensor(image_tensor: torch.Tensor, preprocessing: dict[str, Any]) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    mean = np.array(preprocessing.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
    std = np.array(preprocessing.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)
    image = image * std + mean
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)
    if preprocessing.get("input_space", "RGB") == "BGR":
        image = image[..., ::-1]
    return image


def mask_to_image(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    mask = mask.astype(np.float32)
    max_value = float(mask.max()) if mask.size else 0.0
    if max_value <= 1.0:
        mask = mask * 255.0
    else:
        mask = mask / max(max_value, 1.0) * 255.0
    mask = mask.astype(np.uint8)
    return np.stack([mask, mask, mask], axis=-1)


def get_visual_theme(style: str) -> dict[str, Any]:
    key = str(style or "label").strip().lower()
    return VISUAL_THEMES.get(key, VISUAL_THEMES["label"])


def normalize_mask_indices(mask: np.ndarray) -> np.ndarray:
    mask_array = squeeze_mask_array(mask)
    if mask_array.size == 0:
        return mask_array.astype(np.int32)
    mask_array = np.nan_to_num(mask_array, nan=0.0, posinf=0.0, neginf=0.0)
    mask_array = np.rint(np.clip(mask_array, 0.0, None)).astype(np.int32)
    unique_values = [int(v) for v in np.unique(mask_array) if int(v) > 0]
    if unique_values == [255]:
        binary_mask = np.zeros_like(mask_array, dtype=np.int32)
        binary_mask[mask_array > 0] = 1
        return binary_mask
    return mask_array


def build_visual_palette(style: str) -> np.ndarray:
    theme = get_visual_theme(style)
    palette = [tuple((0, 0, 0))]
    palette.extend(tuple(int(channel) for channel in color) for color in list(theme.get("palette") or []))
    return np.asarray(palette, dtype=np.uint8)


def expand_binary_mask(mask: np.ndarray) -> np.ndarray:
    expanded = mask.copy()
    expanded[1:, :] |= mask[:-1, :]
    expanded[:-1, :] |= mask[1:, :]
    expanded[:, 1:] |= mask[:, :-1]
    expanded[:, :-1] |= mask[:, 1:]
    expanded[1:, 1:] |= mask[:-1, :-1]
    expanded[:-1, :-1] |= mask[1:, 1:]
    expanded[1:, :-1] |= mask[:-1, 1:]
    expanded[:-1, 1:] |= mask[1:, :-1]
    return expanded


def compute_mask_outline(mask_indices: np.ndarray, thickness: int = 2) -> np.ndarray:
    if mask_indices.size == 0:
        return np.zeros(mask_indices.shape, dtype=bool)
    outline = np.zeros(mask_indices.shape, dtype=bool)
    vertical_diff = mask_indices[:-1, :] != mask_indices[1:, :]
    horizontal_diff = mask_indices[:, :-1] != mask_indices[:, 1:]
    outline[:-1, :] |= vertical_diff
    outline[1:, :] |= vertical_diff
    outline[:, :-1] |= horizontal_diff
    outline[:, 1:] |= horizontal_diff
    outline &= mask_indices > 0
    for _ in range(max(int(thickness or 1) - 1, 0)):
        outline = expand_binary_mask(outline) & (mask_indices > 0)
    return outline


def ensure_rgb_image(image: np.ndarray) -> np.ndarray:
    base = np.asarray(image, dtype=np.uint8)
    if base.ndim == 2:
        base = np.stack([base, base, base], axis=-1)
    if base.ndim == 3 and base.shape[-1] == 1:
        base = np.repeat(base, 3, axis=-1)
    return base


def save_image_grid(tiles: list[np.ndarray], columns: int, output_path: str | Path) -> None:
    if not tiles:
        return
    height, width = tiles[0].shape[:2]
    if width >= 900:
        columns = min(max(int(columns or 1), 1), max(1, min(len(tiles), 2)))
    elif width >= 560:
        columns = min(max(int(columns or 1), 1), max(1, min(len(tiles), 3)))
    else:
        columns = min(max(int(columns or 1), 1), max(1, min(len(tiles), 4)))
    rows = (len(tiles) + columns - 1) // columns
    padding = 18
    margin = 24
    canvas_height = rows * height + max(rows - 1, 0) * padding + margin * 2
    canvas_width = columns * width + max(columns - 1, 0) * padding + margin * 2
    canvas = np.full((canvas_height, canvas_width, 3), (241, 245, 249), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        column = index % columns
        top = margin + row * (height + padding)
        left = margin + column * (width + padding)
        canvas[top : top + height, left : left + width] = tile
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    Image.fromarray(canvas).save(output_path)


VALID_PAD_ALIGNS = ("center", "top_left")


def pad_image(
    image: Image.Image,
    size: tuple[int, int],
    fill: int = 0,
    align: str = "center",
) -> tuple[Image.Image, dict[str, int]]:
    """不缩放，直接将原图放到目标尺寸的画布上，不足处用 fill 填充。

    若原图某一边比目标大，则在该边裁剪以放入画布（纯填充无法容纳时的兜底）。

    Args:
        image: 输入 PIL 图像（保持原始分辨率，不缩放）
        size: 目标尺寸 (height, width)
        fill: 填充值，图像填 0（黑），mask 填 0（背景类）
        align: 对齐方式，"center" 居中放置，"top_left" 放在左上角

    Returns:
        padded: 填充后的 PIL 图像，尺寸为 (width, height)
        pad_info: 还原所需元信息 {pad_left, pad_top, src_left, src_top, copy_w, copy_h}
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
    """去除填充并还原到原图尺寸，不缩放。

    从模型尺寸的输出中取出有效区域，放回原图对应位置；被裁掉的边界
    区域没有预测值，保持 0。

    Args:
        mask: 模型尺寸上的 mask/probability，uint8
        original_size: 原图尺寸 (width, height)，与 PIL Image.size 一致
        pad_info: pad_image 返回的元信息
    """
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


def squeeze_mask_array(mask: np.ndarray) -> np.ndarray:
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        mask_array = mask_array[0]
    elif mask_array.ndim == 3 and mask_array.shape[-1] == 1:
        mask_array = mask_array[..., 0]
    elif mask_array.ndim == 3:
        mask_array = np.argmax(mask_array, axis=0)
    return mask_array.astype(np.float32)


def colorize_mask(mask: np.ndarray, style: str = "label") -> np.ndarray:
    mask_indices = normalize_mask_indices(mask)
    if mask_indices.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    max_index = int(mask_indices.max()) if mask_indices.size else 0
    palette = build_visual_palette(style)
    if max_index >= len(palette):
        extras = []
        base_colors = palette[1:] if len(palette) > 1 else np.asarray([(255, 255, 255)], dtype=np.uint8)
        for extra_index in range(max_index - len(palette) + 1):
            extras.append(base_colors[extra_index % len(base_colors)])
        palette = np.concatenate([palette, np.asarray(extras, dtype=np.uint8)], axis=0)
    colored = palette[np.clip(mask_indices, 0, len(palette) - 1)]
    colored[mask_indices <= 0] = 0
    return colored.astype(np.uint8)


def render_mask_preview(mask: np.ndarray, style: str = "label") -> np.ndarray:
    mask_indices = normalize_mask_indices(mask)
    if mask_indices.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    theme = get_visual_theme(style)
    preview = np.full((*mask_indices.shape, 3), theme["preview_background"], dtype=np.uint8)
    colored_mask = colorize_mask(mask_indices, style=style)
    positive = mask_indices > 0
    if np.any(positive):
        preview[positive] = colored_mask[positive]
        outline = compute_mask_outline(mask_indices, thickness=2)
        outline_color = np.clip(colored_mask.astype(np.int16) + 35, 0, 255).astype(np.uint8)
        preview[outline] = outline_color[outline]
    return preview


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.42, style: str = "label") -> np.ndarray:
    base = ensure_rgb_image(image)
    mask_indices = normalize_mask_indices(mask)
    colored_mask = colorize_mask(mask_indices, style=style)
    if colored_mask.shape[:2] != base.shape[:2]:
        return base
    overlay = base.astype(np.float32)
    positive = mask_indices > 0
    if np.any(positive):
        overlay[positive] = overlay[positive] * (1.0 - float(alpha)) + colored_mask[positive].astype(np.float32) * float(alpha)
        outline = compute_mask_outline(mask_indices, thickness=2)
        outline_color = np.clip(colored_mask.astype(np.int16) + 45, 0, 255).astype(np.uint8)
        overlay[outline] = outline_color[outline]
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def compose_visual_strip(image: np.ndarray, mask: np.ndarray, style: str = "label", panels: tuple[str, ...] = ("image", "overlay")) -> np.ndarray:
    theme = get_visual_theme(style)
    base = ensure_rgb_image(image)
    overlay = overlay_mask_on_image(base, mask, style=style)
    preview = render_mask_preview(mask, style=style)
    panel_lookup = {
        "image": base,
        "overlay": overlay,
        "mask": preview if preview.shape[:2] == base.shape[:2] else np.zeros_like(base),
    }
    selected_panels = [ensure_rgb_image(panel_lookup[name]) for name in list(panels or ()) if name in panel_lookup]
    if not selected_panels:
        return overlay
    if len(selected_panels) == 1:
        return selected_panels[0].astype(np.uint8)

    panel_gap = 10
    strip_height = max(panel.shape[0] for panel in selected_panels)
    strip_width = sum(panel.shape[1] for panel in selected_panels) + panel_gap * max(len(selected_panels) - 1, 0)
    strip = np.full((strip_height, strip_width, 3), theme["card_background"], dtype=np.uint8)
    cursor = 0
    for panel in selected_panels:
        height, width = panel.shape[:2]
        top = max((strip_height - height) // 2, 0)
        strip[top : top + height, cursor : cursor + width] = panel
        cursor += width + panel_gap
    return strip.astype(np.uint8)


def compose_visual_card(image: np.ndarray, mask: np.ndarray, style: str = "label", title: str | None = None) -> np.ndarray:
    theme = get_visual_theme(style)
    base = ensure_rgb_image(image)
    overlay = overlay_mask_on_image(base, mask, style=style)
    preview = render_mask_preview(mask, style=style)
    font = ImageFont.load_default()

    overlay_image = Image.fromarray(overlay)
    if preview.size > 0:
        preview_height = max(72, overlay_image.height // 3)
        preview_width = max(72, int(round(float(overlay_image.width) * float(preview_height) / max(float(overlay_image.height), 1.0))))
        preview_width = min(preview_width, max(72, overlay_image.width // 2))
        preview_image = Image.fromarray(preview).resize((preview_width, preview_height), Image.Resampling.NEAREST)
        inset_pad = max(10, min(overlay_image.width, overlay_image.height) // 30)
        inset_left = max(inset_pad, overlay_image.width - preview_width - inset_pad)
        inset_top = max(inset_pad, overlay_image.height - preview_height - inset_pad)
        overlay_image.paste(preview_image, (inset_left, inset_top))
        overlay_draw = ImageDraw.Draw(overlay_image)
        overlay_draw.rectangle(
            [inset_left - 2, inset_top - 2, inset_left + preview_width + 1, inset_top + preview_height + 1],
            outline=(255, 255, 255),
            width=2,
        )
        tag_top = max(0, inset_top - 18)
        overlay_draw.rectangle([inset_left, tag_top, inset_left + 46, tag_top + 18], fill=theme["accent"])
        overlay_draw.text((inset_left + 6, tag_top + 3), "Mask", fill=(255, 255, 255), font=font)

    panel_gap = 14
    outer_pad = 14
    header_height = 34
    card_width = outer_pad * 2 + base.shape[1] * 2 + panel_gap
    card_height = outer_pad * 2 + header_height + base.shape[0]
    card = Image.new("RGB", (card_width, card_height), theme["card_background"])
    draw = ImageDraw.Draw(card)
    draw.rectangle([0, 0, card_width - 1, card_height - 1], outline=theme["border"], width=1)
    draw.rectangle([0, 0, card_width - 1, 5], fill=theme["accent"])
    draw.text((outer_pad, 11), str(title or "Visualization"), fill=theme["text"], font=font)

    panel_top = outer_pad + header_height
    left_panel = (outer_pad, panel_top)
    right_panel = (outer_pad + base.shape[1] + panel_gap, panel_top)
    draw.text((left_panel[0], panel_top - 18), "Image", fill=theme["subtle_text"], font=font)
    draw.text((right_panel[0], panel_top - 18), "Overlay", fill=theme["subtle_text"], font=font)
    card.paste(Image.fromarray(base), left_panel)
    card.paste(overlay_image, right_panel)
    draw.rectangle(
        [left_panel[0] - 1, left_panel[1] - 1, left_panel[0] + base.shape[1], left_panel[1] + base.shape[0]],
        outline=theme["border"],
        width=1,
    )
    draw.rectangle(
        [right_panel[0] - 1, right_panel[1] - 1, right_panel[0] + base.shape[1], right_panel[1] + base.shape[0]],
        outline=theme["border"],
        width=1,
    )
    return np.asarray(card, dtype=np.uint8)


def save_overlay_batch_visualization(
    images: torch.Tensor,
    masks: torch.Tensor | np.ndarray,
    preprocessing: dict[str, Any],
    output_path: str | Path,
    max_items: int,
    style: str = "label",
    title: str | None = None,
    panels: tuple[str, ...] = ("image", "overlay"),
    show_text: bool = True,
) -> None:
    if isinstance(masks, torch.Tensor):
        mask_array = masks.detach().cpu().numpy()
    else:
        mask_array = np.asarray(masks)
    tiles = []
    count = min(max_items, len(images), len(mask_array))
    for index in range(count):
        image = denormalize_image_tensor(images[index], preprocessing)
        card_title = str(title or str(style or "visualization").replace("_", " ").title())
        if show_text:
            tiles.append(
                compose_visual_card(
                    image,
                    mask_array[index],
                    style=style,
                    title=f"{str(card_title)} · Sample {index + 1:02d}",
                )
            )
        else:
            tiles.append(
                compose_visual_strip(
                    image,
                    mask_array[index],
                    style=style,
                    panels=panels,
                )
            )
    save_image_grid(tiles, columns=min(count, 4) or 1, output_path=output_path)


def save_train_batch_visualization(
    images: torch.Tensor,
    masks: torch.Tensor,
    preprocessing: dict[str, Any],
    output_path: str | Path,
    max_items: int,
) -> None:
    save_overlay_batch_visualization(
        images=images,
        masks=masks,
        preprocessing=preprocessing,
        output_path=output_path,
        max_items=max_items,
        style="train",
        panels=("overlay",),
        show_text=False,
    )


def save_mask_batch_visualization(
    masks: torch.Tensor | np.ndarray,
    output_path: str | Path,
    max_items: int,
) -> None:
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    tiles = []
    count = min(max_items, len(masks))
    for index in range(count):
        tiles.append(render_mask_preview(masks[index], style="label"))
    save_image_grid(tiles, columns=min(count, 4) or 1, output_path=output_path)


class SegmentationTxtDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        split_txt: str | Path,
        image_size: tuple[int, int],
        num_classes: int,
        preprocessing: dict[str, Any],
        mask_values: list[int] | None = None,
        transform=None,
        pad: bool = False,
        pad_align: str = "center",
    ):
        self.samples = collect_split_pairs(images_dir, masks_dir, split_txt)
        self.height, self.width = image_size
        self.num_classes = num_classes
        self.preprocessing = preprocessing
        self.mask_values = [int(x) for x in list(mask_values or [])] if mask_values else []
        self.transform = transform
        self.pad = bool(pad)
        self.pad_align = str(pad_align or "center").strip().lower()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        elif self.pad:
            image, _ = pad_image(image, (self.height, self.width), fill=0, align=self.pad_align)
            mask, _ = pad_image(mask, (self.height, self.width), fill=0, align=self.pad_align)
        else:
            image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
            mask = mask.resize((self.width, self.height), Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32)
        image_array = preprocess_image_array(image_array, self.preprocessing)
        image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1))).float()

        mask_array = np.asarray(mask, dtype=np.int64)
        if self.num_classes == 1:
            mask_array = (mask_array > 0).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()
        else:
            if self.mask_values:
                indexed_mask = np.zeros(mask_array.shape, dtype=np.int64)
                for class_index, raw_value in enumerate(self.mask_values):
                    indexed_mask[mask_array == int(raw_value)] = int(class_index)
                mask_array = indexed_mask
            mask_tensor = torch.from_numpy(mask_array).long()

        return image_tensor, mask_tensor


def build_model(
    arch: str,
    encoder_name: str,
    encoder_weights: str | None,
    in_channels: int,
    num_classes: int,
    encoder_depth: int = 5,
    encoder_output_stride: int = 16,
    decoder_channels: tuple[int, ...] | None = None,
) -> torch.nn.Module:
    normalized_arch = normalize_arch_name(arch)
    normalized_encoder_name = str(encoder_name or "").strip().lower()
    if normalized_arch != "Unet" and normalized_encoder_name.startswith("tu-convnextv2_"):
        raise ValueError(
            f"Encoder '{encoder_name}' is currently only supported with Unet in this segmentation pipeline. "
            f"Please switch arch to 'Unet' or choose another encoder for '{normalized_arch}'."
        )
    normalized_weights = resolve_encoder_weights(encoder_name, encoder_weights)
    model_kwargs: dict[str, Any] = {"encoder_depth": encoder_depth}

    if normalized_arch in {"Unet", "UnetPlusPlus"}:
        selected_decoder_channels = decoder_channels or DEFAULT_DECODER_CHANNELS[:encoder_depth]
        model_kwargs["decoder_channels"] = selected_decoder_channels
    elif normalized_arch == "DeepLabV3":
        model_kwargs["encoder_output_stride"] = encoder_output_stride

    return smp.create_model(
        normalized_arch,
        encoder_name=encoder_name,
        encoder_weights=normalized_weights,
        in_channels=in_channels,
        classes=1 if num_classes == 1 else num_classes,
        **model_kwargs,
    )


def create_loss_function(num_classes: int):
    if num_classes == 1:
        dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        bce_loss = smp.losses.SoftBCEWithLogitsLoss()

        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return 0.5 * dice_loss(logits, targets) + 0.5 * bce_loss(logits, targets)

        return loss_fn

    dice_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
    ce_loss = torch.nn.CrossEntropyLoss()

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * dice_loss(logits, targets) + 0.5 * ce_loss(logits, targets)

    return loss_fn


@torch.no_grad()
def compute_batch_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    threshold: float,
) -> float:
    if num_classes == 1:
        probabilities = torch.sigmoid(logits).squeeze(1)
        predictions = (probabilities > threshold).long()
        target_labels = targets.squeeze(1).long()
        tp, fp, fn, tn = smp.metrics.get_stats(predictions, target_labels, mode="binary")
        score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        return float(score.item())

    predictions = torch.argmax(logits, dim=1)
    tp, fp, fn, tn = smp.metrics.get_stats(
        predictions,
        targets.long(),
        mode="multiclass",
        num_classes=num_classes,
    )
    score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    return float(score.item())


@torch.no_grad()
def compute_mask_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    threshold: int = 127,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """使用 smp.metrics 计算 IoU（与训练验证一致）。

    返回 (iou, tp, fp, fn, tn)，其中 tp/fp/fn/tn 为 smp.metrics.get_stats
    的原始输出，可累加后重新计算 Global IoU。
    """
    if num_classes == 1:
        predictions = torch.from_numpy((pred > threshold).astype(np.int64)).unsqueeze(0)
        target_labels = torch.from_numpy((gt > threshold).astype(np.int64)).unsqueeze(0)
        tp, fp, fn, tn = smp.metrics.get_stats(predictions, target_labels, mode="binary")
    else:
        predictions = torch.from_numpy(pred.astype(np.int64)).unsqueeze(0)
        target_labels = torch.from_numpy(gt.astype(np.int64)).unsqueeze(0)
        tp, fp, fn, tn = smp.metrics.get_stats(
            predictions, target_labels, mode="multiclass", num_classes=num_classes,
        )
    iou = float(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item())
    return iou, tp, fp, fn, tn


@torch.no_grad()
def predict_mask_from_logits(
    logits: torch.Tensor,
    num_classes: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if num_classes == 1:
        probabilities = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
        mask = (probabilities > threshold).astype(np.uint8) * 255
        probability_image = np.clip(probabilities * 255.0, 0, 255).astype(np.uint8)
        return mask, probability_image

    probabilities = torch.softmax(logits, dim=1)
    mask = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return mask, mask


def load_image_for_inference(
    image_path: str | Path,
    image_size: tuple[int, int],
    preprocessing: dict[str, Any],
    pad: bool = False,
    pad_align: str = "center",
) -> tuple[torch.Tensor, tuple[int, int], dict[str, int] | None]:
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    pad_info: dict[str, int] | None = None
    if pad:
        resized, pad_info = pad_image(image, image_size, fill=0, align=pad_align)
    else:
        resized = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32)
    image_array = preprocess_image_array(image_array, preprocessing)
    image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1))).float().unsqueeze(0)
    return image_tensor, original_size, pad_info


def resize_mask_to_original(mask: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask)
    image = image.resize(original_size, Image.Resampling.NEAREST)
    return np.asarray(image)


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
    """将分割 mask 以半透明颜色叠加在原图上保存。

    Args:
        image_path: 原始图像路径
        mask: 分割结果，二分类为 0/255 uint8，多分类为类别索引 uint8
        output_path: 叠加图保存路径
        num_classes: 类别数，1 表示二分类
        color: 二分类前景颜色 RGB，默认玫瑰红
        alpha: 叠加透明度，范围 (0, 1)，越大颜色越浓
        metrics: 可选的精度指标字典，非 None 时将指标绘制在图片左上角
    """
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


def save_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    model_config: dict[str, Any],
    image_size: tuple[int, int],
    threshold: float,
    preprocessing: dict[str, Any],
    metrics: dict[str, Any],
    mask_values: list[int] | None = None,
    classes: list[str] | None = None,
    single_cls: bool = False,
    num_classes: int | None = None,
    pad: bool = False,
    pad_align: str = "center",
) -> None:
    model_to_save = model.module if hasattr(model, "module") else model
    payload = {
        "model_state_dict": model_to_save.state_dict(),
        "model_config": model_config,
        "image_size": list(image_size),
        "threshold": threshold,
        "preprocessing": preprocessing,
        "metrics": metrics,
        "mask_values": [int(x) for x in list(mask_values or [])] if mask_values else [],
        "classes": [str(x) for x in list(classes or []) if str(x or "").strip()],
        "single_cls": bool(single_cls),
        "num_classes": int(num_classes) if num_classes is not None else None,
        "pad": bool(pad),
        "pad_align": str(pad_align or "center"),
    }
    checkpoint_path = Path(checkpoint_path)
    ensure_dir(checkpoint_path.parent)
    torch.save(payload, checkpoint_path)


def load_checkpoint(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(checkpoint_path), map_location=map_location)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device,
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


def sahi_predict(
    model: torch.nn.Module,
    image_path: str | Path,
    crop_size: tuple[int, int],
    model_size: tuple[int, int],
    preprocessing: dict[str, Any],
    num_classes: int,
    threshold: float,
    overlap_ratio: float = 0.2,
    device: torch.device | str = "cpu",
    pad: bool = False,
    pad_align: str = "center",
) -> tuple[np.ndarray, np.ndarray]:
    """SAHI 滑窗推理：从原图按 crop_size 裁剪切片，送入模型后推理，加权融合还原至原图尺寸。

    当 pad=True 时，每个裁片按训练时的填充方式放到 model_size 画布上（不缩放），
    与训练预处理保持一致，避免位移；否则将裁片 resize 到 model_size。

    Args:
        model: 已加载的分割模型（eval 模式）
        image_path: 输入图像路径
        crop_size: 从原图裁剪的窗口大小 (height, width)，决定每次看多大的区域
        model_size: 裁片送入模型的尺寸 (height, width)，通常等于训练时的 image_size
        preprocessing: 预处理参数字典
        num_classes: 类别数（1 表示二分类）
        threshold: 二分类概率阈值
        overlap_ratio: 相邻切片重叠比例，范围 [0, 1)，建议 0.1~0.3
        device: 推理设备
        pad: 是否使用填充模式（与训练时一致），True 时不缩放裁片
        pad_align: 填充对齐方式，"center" 或 "top_left"，需与训练时一致

    Returns:
        mask: 融合后的分割 mask，shape (H, W)，uint8
        probability: 融合后的概率图，shape (H, W)，uint8 (0-255)
    """
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    crop_h, crop_w = crop_size
    model_h, model_w = model_size
    stride_h = max(1, int(crop_h * (1.0 - overlap_ratio)))
    stride_w = max(1, int(crop_w * (1.0 - overlap_ratio)))

    if num_classes == 1:
        accum = np.zeros((orig_h, orig_w), dtype=np.float64)
    else:
        accum = np.zeros((num_classes, orig_h, orig_w), dtype=np.float64)
    weight = np.zeros((orig_h, orig_w), dtype=np.float64)

    y_starts = list(range(0, max(orig_h - crop_h, 0) + 1, stride_h))
    x_starts = list(range(0, max(orig_w - crop_w, 0) + 1, stride_w))
    if not y_starts or y_starts[-1] + crop_h < orig_h:
        y_starts.append(max(orig_h - crop_h, 0))
    if not x_starts or x_starts[-1] + crop_w < orig_w:
        x_starts.append(max(orig_w - crop_w, 0))

    model.eval()
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + crop_h, orig_h)
            x1 = min(x0 + crop_w, orig_w)

            patch = image.crop((x0, y0, x1, y1))
            if pad:
                patch_input, patch_pad_info = pad_image(patch, model_size, fill=0, align=pad_align)
            else:
                patch_input = patch.resize((model_w, model_h), Image.Resampling.BILINEAR)
                patch_pad_info = None

            patch_array = np.asarray(patch_input, dtype=np.float32)
            patch_array = preprocess_image_array(patch_array, preprocessing)
            patch_tensor = (
                torch.from_numpy(np.transpose(patch_array, (2, 0, 1)))
                .float()
                .unsqueeze(0)
                .to(device)
            )

            with torch.inference_mode():
                logits = model(patch_tensor)

            actual_h = y1 - y0
            actual_w = x1 - x0

            if num_classes == 1:
                prob = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
                if patch_pad_info is not None:
                    pl = patch_pad_info["pad_left"]
                    pt = patch_pad_info["pad_top"]
                    cw = patch_pad_info["copy_w"]
                    ch = patch_pad_info["copy_h"]
                    prob_region = prob[pt : pt + ch, pl : pl + cw].astype(np.float64)
                else:
                    prob_region = np.asarray(
                        Image.fromarray(prob).resize((actual_w, actual_h), Image.Resampling.BILINEAR),
                        dtype=np.float64,
                    )
                accum[y0:y1, x0:x1] += prob_region
            else:
                prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                for c in range(num_classes):
                    if patch_pad_info is not None:
                        pl = patch_pad_info["pad_left"]
                        pt = patch_pad_info["pad_top"]
                        cw = patch_pad_info["copy_w"]
                        ch = patch_pad_info["copy_h"]
                        ch_region = prob[c][pt : pt + ch, pl : pl + cw].astype(np.float64)
                    else:
                        ch_region = np.asarray(
                            Image.fromarray(prob[c].astype(np.float32)).resize(
                                (actual_w, actual_h), Image.Resampling.BILINEAR
                            ),
                            dtype=np.float64,
                        )
                    accum[c, y0:y1, x0:x1] += ch_region

            weight[y0:y1, x0:x1] += 1.0

    weight = np.maximum(weight, 1.0)

    if num_classes == 1:
        avg_prob = (accum / weight).astype(np.float32)
        mask = (avg_prob > threshold).astype(np.uint8) * 255
        probability = np.clip(avg_prob * 255.0, 0, 255).astype(np.uint8)
    else:
        avg_prob = (accum / weight[None, :, :]).astype(np.float32)
        mask = np.argmax(avg_prob, axis=0).astype(np.uint8)
        probability = np.clip(np.max(avg_prob, axis=0) * 255.0, 0, 255).astype(np.uint8)

    return mask, probability
