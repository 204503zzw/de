# ONNX 导出 + 推理（独立版）

从 `example/unet_pretrained_backbones_demo_v2/` 剥离出来的 ONNX 导出与推理脚本，
不依赖仓库内其他模块，可整个目录复制到任意环境单独使用：

- `infer_onnxruntime.py` — **纯 ONNXRuntime 推理**，只需 numpy/pillow/onnxruntime，
  完全不需要 torch / segmentation-models-pytorch。
- `export_onnx.py` — 把训练得到的 checkpoint (.pth) 导出为 `.onnx`。这一步**必须**
  依赖 torch + segmentation-models-pytorch（要用模型定义重建网络再导出）。

推理只需要一个已经导出好的 `.onnx` 分割模型和待推理图片即可运行。

## 依赖

推理（轻量，无框架）：

```bash
pip install -r requirements.txt   # numpy / pillow / onnxruntime
```

导出（需要模型框架）：

```bash
pip install -r requirements-export.txt   # torch / torchvision / smp / onnx
```

> 说明：推理端之所以不需要框架，是因为模型结构和权重都已经烘焙进 `.onnx` 文件；
> 只有把 checkpoint 转成 `.onnx` 的导出环节才需要 torch + 模型定义。

## 导出

```bash
python export_onnx.py --checkpoint best.pth --output model.onnx
```

会同时写出一份 `model.json` 兼容元数据（推理脚本并不依赖它）。

## 用法

```bash
python infer_onnxruntime.py \
  --onnx model.onnx \
  --input path/to/images_or_single_image \
  --output-dir outputs
```

`--input` 可以是单张图片，也可以是一个目录（会遍历常见图片格式）。

输出：每张图会写出 `<name>_mask.png`；单类模型或加 `--save-prob` 时还会写出
`<name>_prob.png`。

### 常用参数

- `--imgsz H W`：显式指定输入尺寸（动态输入的 onnx 建议传）。
- `--dynamic --stride 32`：保持原图尺寸，仅填充到 stride 倍数后推理，避免 resize 变形。
- `--pad [--pad-align center|top_left]`：不缩放，直接把原图放到模型尺寸画布并填充黑边。
- `--threshold 0.5`：二值分割阈值。
- `--input-space RGB|BGR`、`--input-range 0 1`、`--mean ...`、`--std ...`：预处理，
  与训练时保持一致（3 通道默认使用 ImageNet 归一化）。
- `--overlay [--overlay-alpha 0.45]`：额外保存把 mask 叠加到原图的可视化结果。
- `--gt-dir DIR [--metrics-output metrics.csv]`：提供 ground-truth mask 目录时计算
  IoU / Dice / Precision / Recall / Accuracy 等指标并可导出 CSV。

输入通道数、类别数会自动从 onnx 的输入/输出 shape 推断。

## 与原脚本的差异

功能与命令行参数完全一致；唯一区别是把原来 `from common import compute_mask_iou`
所依赖的 `smp.metrics` 换成了等价的纯 numpy 实现（micro IoU 语义不变），从而彻底
去掉 `torch` / `smp` 依赖。
