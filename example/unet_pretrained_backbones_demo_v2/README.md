
# unet_pretrained_backbones_demo_v2

`v2` 是一套可以**单独打包、单独交付、单独运行**的语义分割脚本包。

你可以把当前目录整体发给别人，作为一个独立的训练 / 推理 / 导出工具使用，而不必要求对方先理解原始项目结构。

这套脚本支持：

- `Unet`
- `UnetPlusPlus`
- `Segformer`
- `DeepLabV3`
- `resnet18` / `mobilenet_v2` / `efficientnet-*` 等 encoder
- `encoder_depth`
- 单类与多类别语义分割
- 常见训练增强
  - 左右翻转
  - 上下翻转
  - 90° 旋转
  - 平移
  - 缩放
  - 噪声
  - 亮度
  - 对比度
- 保存训练 batch、验证预测、最佳/最后权重、指标结果与曲线文件
- PyTorch 推理
- ONNX 导出
- ONNXRuntime 推理
- 独立验证模块（复现训练时的 val_loss / val_iou）
- 推理时精度统计（IoU、Dice、Precision、Recall、Accuracy）
- 动态推理模式（保持原图尺寸，避免 resize 变形）

## 1. 你拿到这份包后先做什么

建议把当前目录完整保留，不要只复制单个脚本。

一个最常见的可交付形态是：

```text
unet_pretrained_backbones_demo_v2/
  README.md
  train_segmentation.py
  prepare_splits.py
  evaluate.py
  infer_pytorch.py
  export_onnx.py
  infer_onnxruntime.py
  common.py
  augmentations.py
```

后续命令都默认你在**当前 README 所在目录**执行。

这套脚本内部仍保留了一些机器相关的默认路径，只是为了历史兼容；真正交付给别人时，**不要依赖默认值**，请始终显式传参。

## 2. 包内文件说明

- `train_segmentation.py`
  - 训练主脚本
- `prepare_splits.py`
  - 根据 `images/` 与 `masks/` 自动生成 `train.txt`、`val.txt`
- `infer_pytorch.py`
  - 使用 `.pth` checkpoint 做 PyTorch 推理
- `export_onnx.py`
  - 把 `.pth` 导出成 `.onnx`
- `infer_onnxruntime.py`
  - 使用 ONNXRuntime 做推理
- `evaluate.py`
  - 独立验证模块，加载模型 + 验证集计算 val_loss 和 val_iou
- `common.py`
  - 模型构建、数据读取、可视化、checkpoint 读写等公共逻辑
- `augmentations.py`
  - 训练增强实现

## 3. 环境准备

建议为这套脚本单独准备一个 Python 环境，并先安装与你机器匹配的 `torch` / `torchvision`。

当前目录已经附带一份 `requirements.txt`，最简单的安装方式是：

```powershell
pip install -r requirements.txt
```

这份依赖默认包含 CPU 版 `onnxruntime`。

如果你需要 ONNX 导出或 ONNX 推理，这个命令已经足够。

如果你要走 GPU 版 ONNXRuntime，建议把：

- `onnxruntime`

替换成：

```powershell
onnxruntime-gpu
```

也可以直接手动执行：

```powershell
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
```

说明：

- 如果你的 `torch` / `torchvision` 需要和特定 CUDA 版本严格匹配，建议优先按 PyTorch 官方方式安装，再执行 `pip install -r requirements.txt`
- 第一次使用 `imagenet` 预训练权重时，可能会联网下载权重
- 如果不希望联网，可以把 `--encoder-weights` 设为 `none`

## 4. 数据组织方式

推荐数据结构如下：

```text
dataset/
  images/
    0001.jpg
    0002.jpg
  masks/
    0001.png
    0002.png
  splits/
    train.txt
    val.txt
```

其中：

- `images/` 放原图
- `masks/` 放分割 mask
- `train.txt` / `val.txt` 负责定义训练集与验证集

`train.txt` / `val.txt` 每行支持：

- 样本 stem，例如 `0001`
- 文件名，例如 `0001.jpg`
- 相对路径，例如 `train/0001.jpg`

这套脚本也支持较旧的数据组织方式，例如：

```text
images/
  train/
  val/
masks/
  train/
  val/
```

因为脚本会递归扫描 `images/` 与 `masks/`。

## 5. 生成 train.txt / val.txt

如果你还没有划分文件，可以先运行：

```powershell
python prepare_splits.py ^
  --images-dir C:\path\to\dataset\images ^
  --masks-dir C:\path\to\dataset\masks ^
  --output-dir C:\path\to\dataset\splits ^
  --train-ratio 0.8 ^
  --val-ratio 0.2
```

生成后会得到：

- `train.txt`
- `val.txt`

如果你的数据已经手动划分好了，也可以跳过这一步。

## 6. 开始训练

最推荐的训练方式是**显式传入所有路径**：

```powershell
python train_segmentation.py ^
  --images-dir C:\path\to\dataset\images ^
  --masks-dir C:\path\to\dataset\masks ^
  --train-txt C:\path\to\dataset\splits\train.txt ^
  --val-txt C:\path\to\dataset\splits\val.txt ^
  --save-dir C:\path\to\runs
```

默认训练配置大致为：

- `arch=Unet`
- `encoder_name=resnet18`
- `encoder_weights=imagenet`
- `encoder_depth=5`
- `epochs=100`
- `batch_size=8`
- `lr=5e-4`
- `threshold=0.25`

说明：

- `--save-dir` 是运行结果的父目录
- 脚本会在该目录下再创建一个运行子目录
- 不建议直接依赖脚本里保留的历史默认路径

## 7. 主要训练参数说明

- `--arch`
  - `Unet`
  - `UnetPlusPlus`
  - `Segformer`
  - `DeepLabV3`

- `--encoder-name`
  - 例如 `resnet18`
  - `mobilenet_v2`
  - `efficientnet-b0`

- `--encoder-weights`
  - `imagenet`
  - `none`

- `--encoder-depth`
  - 主要用于 `Unet` / `UnetPlusPlus`
  - 一般使用 `3 ~ 5`

- `--encoder-output-stride`
  - 主要用于 `DeepLabV3`
  - 常用 `8` 或 `16`

- `--in-channels`
  - 输入图像通道数，默认 `3`

- `--num-classes`
  - 语义分割类别数
  - 二分类通常为 `1`
  - 多分类按输出通道数训练与推理

- `--force-single-class`
  - 强制按单类掩码训练

- `--height` / `--width`
  - 训练输入尺寸

- `--threshold`
  - 二分类预测阈值
  - 多分类推理走 `argmax`，不依赖该阈值

- `--pad`
  - 启用后不缩放图像，直接将原图放到目标尺寸的画布上，不足处用黑色填充
  - 保持原图像素的原始分辨率，避免缩放带来的失真
  - 若原图某一边比目标大，则在该边裁剪后放入画布

- `--pad-align`
  - 填充时原图的对齐方式，需配合 `--pad` 使用
  - `center`（默认）：原图居中放置，四周填充黑色
  - `top_left`：原图放在左上角，右侧和下方填充黑色

- 增广参数
  - `--hflip-prob`
  - `--vflip-prob`
  - `--rotate90-prob`
  - `--shift-prob`
  - `--max-shift-ratio`
  - `--scale-prob`
  - `--min-scale`
  - `--max-scale`
  - `--noise-prob`
  - `--noise-std`
  - `--brightness-prob`
  - `--brightness-min`
  - `--brightness-max`
  - `--contrast-prob`
  - `--contrast-min`
  - `--contrast-max`

## 8. 常用训练命令示例

### Unet + resnet18

```powershell
python train_segmentation.py --arch Unet --encoder-name resnet18 --encoder-weights imagenet --encoder-depth 5
```

### UnetPlusPlus + resnet18

```powershell
python train_segmentation.py --arch UnetPlusPlus --encoder-name resnet18 --encoder-weights imagenet --encoder-depth 5
```

### Unet + mobilenet_v2

```powershell
python train_segmentation.py --arch Unet --encoder-name mobilenet_v2 --encoder-weights imagenet --encoder-depth 5
```

### Unet + efficientnet-b0

```powershell
python train_segmentation.py --arch Unet --encoder-name efficientnet-b0 --encoder-weights imagenet --encoder-depth 5
```

### Segformer

```powershell
python train_segmentation.py --arch Segformer --encoder-name resnet18 --encoder-weights imagenet --encoder-depth 5
```

### DeepLabV3

```powershell
python train_segmentation.py --arch DeepLabV3 --encoder-name resnet18 --encoder-weights imagenet --encoder-output-stride 16
```

### 不使用预训练权重

```powershell
python train_segmentation.py --encoder-weights none
```

### 使用填充模式训练（不缩放原图）

```powershell
python train_segmentation.py --pad
```

### 填充时原图放在左上角

```powershell
python train_segmentation.py --pad --pad-align top_left
```

### 多类别分割

```powershell
python train_segmentation.py --num-classes 4
```

## 9. 训练输出内容

每次训练会在 `runs/项目名/` 下生成：

```text
runs/
  20260324_120000_unet_resnet18/
    config.json
    results.csv
    results.png
    metric_results.json
    metric_results.csv
    unet_meta.json
    train_batch0.png
    train_batch1.png
    val_batch_label_ep1.png
    val_batch_pred_ep1.png
    val_batch_label_ep50.png
    val_batch_pred_ep50.png
    val_batch_label_ep100.png
    val_batch_pred_ep100.png
    weight/
      best.pth
      last.pth
```

含义如下：

- `config.json`
  - 当前运行的完整配置快照
- `results.csv`
  - 每轮 `train_loss`、`val_loss`、`train_iou`、`val_iou` 等曲线数据
- `results.png`
  - 由 `results.csv` 生成的训练曲线图
- `train_batch0.png`
  - 第 0 个训练 batch 的可视化
- `train_batch1.png`
  - 第 1 个训练 batch 的可视化
- `val_batch_label_epX.png`
  - 第 `X` 轮验证标签图
- `val_batch_pred_epX.png`
  - 第 `X` 轮验证预测图
- `weight/best.pth`
  - 按 `val_iou` 最优保存
- `weight/last.pth`
  - 最后一轮模型
- `unet_meta.json`
  - 如果数据侧提供了类别等元信息，运行目录会同步保存一份
- `metric_results.json`
  - 每轮指标与 summary
- `metric_results.csv`
  - 方便直接用 Excel 看曲线

## 10. PyTorch 推理

```powershell
python infer_pytorch.py --checkpoint C:\path\to\best.pth --input C:\path\to\image_or_dir --output-dir C:\path\to\output
```

输出：

- `xxx_mask.png` — 分割 mask（灰度图）
- `xxx_prob.png` — 概率图（灰度图）
- `xxx_overlay.png` — mask 叠加在原图上的可视化（需加 `--overlay`）

支持：

- 单张图片输入
- 整个文件夹输入

### 10.1 SAHI 滑窗推理（适合高分辨率大图）

SAHI（Slicing Aided Hyper Inference）将大图切成若干带重叠的小块，对每个小块独立推理，再把所有小块的预测概率**加权平均融合**还原成完整 mask。

适用场景：

- 图像分辨率远大于训练尺寸（如 2K / 4K 工业图像）
- 目标较小、整图缩放后细节丢失严重

**基本用法**

```powershell
python infer_pytorch.py ^
  --checkpoint C:\path\to\best.pth ^
  --input C:\path\to\image_or_dir ^
  --output-dir C:\path\to\output ^
  --sahi
```

**自定义切片尺寸**

不指定时默认使用训练时的 `image_size`。

```powershell
python infer_pytorch.py ^
  --checkpoint C:\path\to\best.pth ^
  --input C:\path\to\image_or_dir ^
  --output-dir C:\path\to\output ^
  --sahi ^
  --sahi-crop-size 640 640 ^
  --sahi-model-size 640 640
```

**自定义重叠比例**

重叠比例越大，切片边界处融合越平滑，但切片总数增多、推理变慢；默认 `0.2`。

```powershell
python infer_pytorch.py ^
  --checkpoint C:\path\to\best.pth ^
  --input C:\path\to\image_or_dir ^
  --output-dir C:\path\to\output ^
  --sahi ^
  --sahi-crop-size 512 512 ^
  --sahi-model-size 512 512 ^
  --sahi-overlap 0.25
```

**SAHI 参数说明**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--sahi` | 关闭 | 启用 SAHI 滑窗推理 |
| `--sahi-crop-size H W` | 训练时 image_size | 从原图裁剪的窗口大小（高 宽），决定每次看多大区域 |
| `--sahi-model-size H W` | 训练时 image_size | 裁片 resize 后送入模型的尺寸（高 宽），通常与训练尺寸一致 |
| `--sahi-overlap` | `0.2` | 相邻切片的重叠比例，范围 `[0, 1)`，建议 `0.1 ~ 0.3` |

**切片数量估算**

以图像 `2048×2048`、切片 `512×512`、重叠 `0.2` 为例：

```
stride = 512 × (1 - 0.2) = 409
行数 = ceil((2048 - 512) / 409) + 1 ≈ 6
列数 = 6
总切片数 = 6 × 6 = 36 张
```

注意：

- 切片尺寸不必须等于训练尺寸，但越接近效果越好
- 边缘切片**自动对齐**：最后一个切片的起始位置会回退到 `orig_size - crop_size`，确保切片始终是完整的 `crop_size`，不会出现不足尺寸的裁片；因此边缘处的实际重叠会略大于 `--sahi-overlap`
- 若原图本身小于 `--sahi-crop-size`，退化为单次整图推理（仍会 resize 到 `--sahi-model-size`）
- SAHI 推理输出分辨率与原图一致，无需额外 resize

### 10.2 叠加可视化（--overlay）

加上 `--overlay` 后，每张图会额外保存一张将 mask 以半透明颜色叠加在原图上的可视化结果（`{stem}_overlay.png`）。

```powershell
python infer_pytorch.py ^
  --checkpoint C:\path\to\best.pth ^
  --input C:\path\to\image_or_dir ^
  --output-dir C:\path\to\output ^
  --overlay
```

可通过 `--overlay-alpha` 调整叠加透明度（默认 `0.45`，越大颜色越浓）：

```powershell
python infer_pytorch.py ^
  --checkpoint C:\path\to\best.pth ^
  --input C:\path\to\image_or_dir ^
  --output-dir C:\path\to\output ^
  --overlay ^
  --overlay-alpha 0.5
```

`--overlay` 与 `--sahi` 可以同时使用，叠加图基于最终融合后的 mask 生成。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--overlay` | 关闭 | 额外保存 mask 叠加在原图上的可视化结果 |
| `--overlay-alpha` | `0.45` | 叠加透明度，范围 `(0, 1)`，越大颜色越浓 |

### 10.3 精度统计（--gt-dir）

如果有 Ground Truth mask，可以在推理时同步计算精度指标：

```powershell
python infer_pytorch.py ^
  --checkpoint C:\path\to\best.pth ^
  --input C:\path\to\images ^
  --output-dir C:\path\to\output ^
  --gt-dir C:\path\to\gt_masks
```

每张图会打印 IoU、Dice、Precision、Recall、Accuracy，最后输出 Global（全局像素级）和 Mean（逐图平均）两种汇总。

其中 IoU 使用与训练验证时相同的 `smp.metrics` 计算方式，其他指标使用 numpy 计算。

GT mask 要求：

- 灰度图（单通道）
- 二值分割：前景像素值 = 255（白色），背景 = 0（黑色）
- 多类分割：像素值 = 类别索引（0, 1, 2, ...）
- 文件名需与输入图片同名（扩展名可以不同）

加上 `--metrics-output` 可以将指标保存到 CSV 文件：

```powershell
python infer_pytorch.py ^
  --checkpoint C:\path\to\best.pth ^
  --input C:\path\to\images ^
  --output-dir C:\path\to\output ^
  --gt-dir C:\path\to\gt_masks ^
  --metrics-output C:\path\to\output\metrics.csv
```

同时使用 `--overlay` 和 `--gt-dir` 时，会生成两张 overlay 图：

- `{stem}_overlay.png` — 纯 mask 叠加原图
- `{stem}_overlay_metrics.png` — 带精度指标文字的版本

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--gt-dir` | 无 | Ground truth mask 目录 |
| `--metrics-output` | 无 | 精度指标保存路径（CSV 文件） |

### 10.4 动态推理模式（--dynamic）

默认推理会将图片 resize 到训练时的固定尺寸（如 640×640），可能导致宽高比变形。加上 `--dynamic` 后保持原图尺寸，仅在右下填充最少的像素让宽高对齐到模型步长（默认 32）的倍数，推理后裁回原尺寸。

```powershell
python infer_pytorch.py ^
  --checkpoint C:\path\to\best.pth ^
  --input C:\path\to\image_or_dir ^
  --output-dir C:\path\to\output ^
  --dynamic
```

处理过程（以原图 651×490 为例）：

```
原图 651×490
  → 填充到 672×512（32 的倍数）
  → 模型推理
  → 裁回 651×490
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dynamic` | 关闭 | 保持原图尺寸，仅填充到 stride 的倍数 |
| `--stride` | `32` | 对齐步长，取决于编码器下采样层数（5 层 = 2⁵ = 32） |

## 11. 导出 ONNX

```powershell
python export_onnx.py --checkpoint C:\path\to\best.pth --output C:\path\to\model.onnx
```

导出后会同时得到：

- `model.onnx`
- `model.json`

其中 `model.json` 仍会保留一份兼容元数据，但当前 `infer_onnxruntime.py` 已经不再依赖它。

## 12. ONNXRuntime 推理

```powershell
python infer_onnxruntime.py ^
  --onnx C:\path\to\model.onnx ^
  --input C:\path\to\image_or_dir ^
  --output-dir C:\path\to\output
```

如果训练时使用了填充模式，推理时也需要加上对应参数：

```powershell
python infer_onnxruntime.py ^
  --onnx C:\path\to\model.onnx ^
  --input C:\path\to\image_or_dir ^
  --output-dir C:\path\to\output ^
  --pad ^
  --pad-align top_left
```

注意：`infer_pytorch.py` 会自动从 checkpoint 读取 `pad` 和 `pad_align`，无需手动指定；`infer_onnxruntime.py` 需要手动传入。

默认行为：

- 输入通道数从 ONNX 输入 shape 推断
- 类别数从 ONNX 输出通道数推断
- provider 默认优先 `CUDAExecutionProvider`，不可用时回退 `CPUExecutionProvider`
- 对 3 通道 RGB 输入默认使用 ImageNet 预处理
  - `mean=[0.485, 0.456, 0.406]`
  - `std=[0.229, 0.224, 0.225]`
  - `input-range=0 1`

如果 ONNX 是动态输入，建议显式传入：

```powershell
python infer_onnxruntime.py ^
  --onnx C:\path\to\model.onnx ^
  --input C:\path\to\image_or_dir ^
  --output-dir C:\path\to\output ^
  --imgsz 512 512 ^
  --threshold 0.5 ^
  --save-prob
```

如果训练时使用了自定义预处理，也可以显式覆盖：

- `--input-space`
- `--input-range`
- `--mean`
- `--std`

其中 `--input-range 0 1` 表示先把像素从 `0~255` 缩放到 `0~1`，再做 `mean/std` 标准化，这与 ImageNet 预处理一致。

`infer_onnxruntime.py` 同样支持 `--overlay`、`--overlay-alpha`、`--gt-dir`、`--metrics-output`、`--dynamic` 等，用法与 PyTorch 推理版本相同。

## 13. 独立验证模块（evaluate.py）

`evaluate.py` 是一个独立的验证模块，功能类似训练时的验证循环，但不需要训练过程。它加载模型和验证集数据，计算与训练时完全一致的 `val_loss` 和 `val_iou`。

### 13.1 基本用法

**PyTorch checkpoint（固定尺寸，和训练验证一致）**

```bash
python evaluate.py --checkpoint /path/to/best.pth \
    --images-dir /path/to/images --masks-dir /path/to/masks \
    --val-txt /path/to/val.txt
```

**ONNX 模型（需手动指定参数）**

```bash
python evaluate.py --onnx /path/to/model.onnx \
    --images-dir /path/to/images --masks-dir /path/to/masks \
    --val-txt /path/to/val.txt \
    --num-classes 1 --imgsz 640 640
```

ONNX 模式下只计算 `val_iou`（没有 loss 函数），`val_loss` 显示为 nan。

### 13.2 三种验证方式

| 方式 | 参数 | 处理方式 | 适用场景 |
|------|------|----------|----------|
| 拉伸（默认） | 无额外参数 | resize 到固定 `image_size` | 和训练验证一致 |
| 填充 | `--pad` | 原图放到固定画布，不缩放 | 避免 resize 变形 |
| 动态 | `--dynamic` | 保持原图尺寸，补到 32 倍数 | 原始分辨率验证 |

**动态模式**

```bash
python evaluate.py --checkpoint /path/to/best.pth \
    --images-dir /path/to/images --masks-dir /path/to/masks \
    --val-txt /path/to/val.txt --dynamic
```

动态模式保持原图尺寸，仅右下填充到 stride（默认 32）的倍数，逐张推理。无法 batch 处理（因为每张图尺寸不同），所以会比固定尺寸模式慢。

### 13.3 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--checkpoint` | — | PyTorch checkpoint 路径（与 `--onnx` 二选一） |
| `--onnx` | — | ONNX 模型路径（与 `--checkpoint` 二选一） |
| `--images-dir` | — | 图片目录 |
| `--masks-dir` | — | GT mask 目录 |
| `--val-txt` | — | 验证集文件列表 |
| `--imgsz H W` | 从 checkpoint 读取 | 输入图像尺寸，可覆盖 checkpoint 中保存的值 |
| `--batch-size` | `8` | 验证 batch size（固定尺寸模式） |
| `--threshold` | 从 checkpoint 读取 | 二值分割阈值 |
| `--pad` | 关闭 | 使用填充模式（默认不开启） |
| `--pad-align` | `center` | 填充对齐方式 |
| `--dynamic` | 关闭 | 动态推理模式 |
| `--stride` | `32` | 动态推理时的对齐步长 |
| `--amp` | 关闭 | 启用混合精度推理 |
| `--output-dir` | 无 | 输出目录，保存 JSON 结果 |

### 13.4 输出示例

```
============================================================
Evaluation Summary (860 images, 22.1s)
============================================================
val_loss=0.1234  val_iou=0.8567
============================================================
```

如果指定了 `--output-dir`，结果会保存到 `eval_results.json`。

## 14. 打包给别人时的建议

如果你准备把这套脚本直接发给别人，建议至少包含：

- 当前整个目录
- 本 README
- 一份依赖安装说明
- 一个数据目录示例
- 如果只是推理用途，额外附带 `best.pth`

如果对方只是想使用模型，而不是继续训练，通常更推荐交付：

- `infer_pytorch.py`
- `common.py`
- `best.pth`
- 本 README

这样会比交付完整训练链路更轻量。

这套脚本是**独立源码脚本包**，但不是“双击即用”的图形化成品；对方仍需要：

- 自己准备 Python 环境
- 自己安装依赖
- 自己提供数据路径或模型路径

## 15. 常见提醒

- 不要依赖脚本内保留的历史默认路径
- 训练时建议总是显式传入：
  - `--images-dir`
  - `--masks-dir`
  - `--train-txt`
  - `--val-txt`
  - `--save-dir`
- `export_onnx.py` 会额外写出 `model.json`，但当前 `infer_onnxruntime.py` 并不依赖它
- 如果训练时使用 `imagenet` 预训练权重，第一次运行可能需要联网下载
