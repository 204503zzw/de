"""把训练得到的 checkpoint (.pth) 导出为 ONNX。

与推理脚本不同，导出**必须**依赖 torch + segmentation-models-pytorch，因为要用模型
定义重建网络结构再导出。此文件把原工程 `common.py` 里导出所需的少量函数内联进来，
因此不依赖仓库内其他模块，可整目录复制使用。

依赖见 `requirements-export.txt`。
"""

import argparse
import json
from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch

DEFAULT_DECODER_CHANNELS = (256, 128, 64, 32, 16)
ARCH_ALIASES = {
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "unet++": "UnetPlusPlus",
    "segformer": "Segformer",
    "deeplabv3": "DeepLabV3",
}


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized.lower() in {"", "none", "null"}:
        return None
    return normalized


def resolve_encoder_weights(encoder_name: str, encoder_weights: Any) -> str | bool | None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    image_size = tuple(checkpoint["image_size"])
    model_config = checkpoint["model_config"]

    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    dummy_input = torch.randn(
        1,
        int(model_config["in_channels"]),
        int(image_size[0]),
        int(image_size[1]),
        device=device,
    )

    dynamic_axes = {0: "batch_size", 2: "height", 3: "width"}
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": dynamic_axes, "output": dynamic_axes},
    )

    metadata = {
        "onnx_path": str(output_path),
        "image_size": checkpoint["image_size"],
        "pad": bool(checkpoint.get("pad", False)),
        "pad_align": str(checkpoint.get("pad_align", "center") or "center"),
        "threshold": checkpoint.get("threshold", 0.5),
        "preprocessing": checkpoint["preprocessing"],
        "model_config": model_config,
    }
    metadata_path = output_path.with_suffix(".json")
    save_json(metadata_path, metadata)

    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"Exported and validated ONNX model: {output_path}")
    except ImportError:
        print(f"Exported ONNX model: {output_path}")
        print("Install onnx if you also want automatic validation.")

    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
