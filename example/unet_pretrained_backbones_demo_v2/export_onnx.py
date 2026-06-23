import argparse
from pathlib import Path

import torch

from common import ensure_dir, load_model_from_checkpoint, save_json


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
