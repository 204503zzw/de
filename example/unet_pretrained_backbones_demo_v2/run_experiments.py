"""256 切片有效性验证实验编排器。

根据前次讨论的 7 个验证维度，自动编排以下实验:

  1. 公平对比：SAHI 原图推理 + GT 对比
  2. 消融实验：切片尺寸 (256 / 384 / 512)
  3. 切片策略消融：训练 overlap + 推理 overlap
  4. 验证集可靠性：K-fold 交叉验证 / 多种子
  5. 原图训练公平基线：统一验证集
  6. 最终测试集评估：独立 test set
  7. 定性分析：可视化对比

用法::

    # 运行所有实验（需指定数据路径和已有 checkpoint）
    python run_experiments.py \\
        --images-dir /data/images \\
        --masks-dir /data/masks \\
        --output-dir /data/experiments \\
        --checkpoint-256 /path/to/256_model/best.pth \\
        --checkpoint-640 /path/to/640_model/best.pth \\
        --experiments all

    # 只运行实验 1（SAHI 对比）
    python run_experiments.py \\
        --images-dir /data/images \\
        --masks-dir /data/masks \\
        --output-dir /data/experiments \\
        --checkpoint-256 /path/to/best.pth \\
        --experiments 1

    # 运行实验 2（切片尺寸消融，需要训练）
    python run_experiments.py \\
        --images-dir /data/images \\
        --masks-dir /data/masks \\
        --train-txt /data/train.txt \\
        --val-txt /data/val.txt \\
        --output-dir /data/experiments \\
        --experiments 2

    # 运行实验 4（K-fold 交叉验证）
    python run_experiments.py \\
        --images-dir /data/images \\
        --masks-dir /data/masks \\
        --output-dir /data/experiments \\
        --experiments 4 --kfold 5
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="256 切片有效性验证实验编排器"
    )
    # 数据路径
    parser.add_argument("--images-dir", type=str, required=True,
                        help="原始图像目录")
    parser.add_argument("--masks-dir", type=str, required=True,
                        help="原始 mask 目录")
    parser.add_argument("--train-txt", type=str, default=None,
                        help="训练集划分文件（实验 2/3/4/5 需要）")
    parser.add_argument("--val-txt", type=str, default=None,
                        help="验证集划分文件")
    parser.add_argument("--test-txt", type=str, default=None,
                        help="测试集划分文件（实验 6 需要）")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="实验输出根目录")

    # 已有模型
    parser.add_argument("--checkpoint-256", type=str, default=None,
                        help="256 切片训练的 best checkpoint（实验 1/3/6/7 使用）")
    parser.add_argument("--checkpoint-640", type=str, default=None,
                        help="640 原图训练的 best checkpoint（实验 1/5/7 使用）")

    # 实验选择
    parser.add_argument("--experiments", type=str, default="all",
                        help="要运行的实验编号，逗号分隔或 all（默认 all）")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练 epoch 数（默认 100）")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="训练 batch size（默认 8）")
    parser.add_argument("--encoder-name", type=str, default="resnet18",
                        help="编码器名称（默认 resnet18）")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备（默认 auto）")
    parser.add_argument("--amp", action="store_true",
                        help="启用混合精度训练")

    # 实验参数
    parser.add_argument("--tile-sizes", type=str, default="256,384,512",
                        help="实验 2 的切片尺寸列表，逗号分隔（默认 256,384,512）")
    parser.add_argument("--overlaps", type=str, default="0.0,0.25,0.5",
                        help="实验 3 的训练 overlap 比例列表（默认 0.0,0.25,0.5）")
    parser.add_argument("--sahi-overlaps", type=str, default="0.2,0.3,0.5",
                        help="实验 3 的推理 SAHI overlap 比例列表（默认 0.2,0.3,0.5）")
    parser.add_argument("--kfold", type=int, default=5,
                        help="实验 4 的折数（默认 5）")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="实验 4 多种子的种子列表（默认 42,123,456）")
    parser.add_argument("--baseline-sizes", type=str, default="320,640",
                        help="实验 5 原图训练尺寸列表（默认 320,640）")

    return parser.parse_args()


def run_command(cmd: list[str], description: str, dry_run: bool = False) -> int:
    """运行子进程命令并打印状态。"""
    print(f"\n{'=' * 70}")
    print(f"[EXPERIMENT] {description}")
    print(f"[COMMAND] {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    if dry_run:
        print("[DRY RUN] 跳过执行")
        return 0

    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"[WARNING] 命令退出码: {result.returncode}")
    return result.returncode


def ensure_split_files(args: argparse.Namespace) -> None:
    """检查必要的划分文件是否存在。"""
    if args.train_txt and not Path(args.train_txt).is_file():
        raise FileNotFoundError(f"训练集划分文件不存在: {args.train_txt}")
    if args.val_txt and not Path(args.val_txt).is_file():
        raise FileNotFoundError(f"验证集划分文件不存在: {args.val_txt}")
    if args.test_txt and not Path(args.test_txt).is_file():
        raise FileNotFoundError(f"测试集划分文件不存在: {args.test_txt}")


def save_experiment_summary(output_dir: Path, experiment_id: str, info: dict) -> None:
    """保存单个实验的配置和结果摘要。"""
    summary_path = output_dir / f"experiment_{experiment_id}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 实验 1: SAHI 原图推理 + GT 对比
# ---------------------------------------------------------------------------

def run_experiment_1(args: argparse.Namespace) -> None:
    """公平对比：SAHI 原图推理 + GT 对比"""
    exp_dir = Path(args.output_dir) / "exp1_sahi_fullimage_eval"
    os.makedirs(exp_dir, exist_ok=True)

    val_txt = args.val_txt
    if not val_txt:
        print("[SKIP] 实验 1 需要 --val-txt 参数")
        return

    checkpoints = []
    if args.checkpoint_256:
        checkpoints.append(("256_tile_model", args.checkpoint_256))
    if args.checkpoint_640:
        checkpoints.append(("640_orig_model", args.checkpoint_640))

    if not checkpoints:
        print("[SKIP] 实验 1 需要至少一个 checkpoint (--checkpoint-256 或 --checkpoint-640)")
        return

    for model_name, checkpoint_path in checkpoints:
        model_output = exp_dir / model_name
        os.makedirs(model_output, exist_ok=True)

        # SAHI 推理 + GT 评估
        cmd = [
            sys.executable, str(SCRIPT_DIR / "infer_pytorch.py"),
            "--checkpoint", checkpoint_path,
            "--input", args.images_dir,
            "--output-dir", str(model_output / "sahi_infer"),
            "--sahi",
            "--sahi-overlap", "0.2",
            "--gt-dir", args.masks_dir,
            "--metrics-output", str(model_output / "sahi_metrics.csv"),
            "--overlay",
            "--device", args.device,
        ]
        run_command(cmd, f"实验 1: SAHI 原图推理 - {model_name}")

        # 直接推理（不用 SAHI）作为对照
        cmd_direct = [
            sys.executable, str(SCRIPT_DIR / "infer_pytorch.py"),
            "--checkpoint", checkpoint_path,
            "--input", args.images_dir,
            "--output-dir", str(model_output / "direct_infer"),
            "--gt-dir", args.masks_dir,
            "--metrics-output", str(model_output / "direct_metrics.csv"),
            "--overlay",
            "--device", args.device,
        ]
        run_command(cmd_direct, f"实验 1: 直接推理对照 - {model_name}")

    save_experiment_summary(exp_dir, "1", {
        "description": "SAHI 原图推理 + GT 对比",
        "models": [name for name, _ in checkpoints],
    })


# ---------------------------------------------------------------------------
# 实验 2: 切片尺寸消融
# ---------------------------------------------------------------------------

def run_experiment_2(args: argparse.Namespace) -> None:
    """消融实验：切片尺寸 (256 / 384 / 512)"""
    exp_dir = Path(args.output_dir) / "exp2_tile_size_ablation"
    os.makedirs(exp_dir, exist_ok=True)

    if not args.train_txt or not args.val_txt:
        print("[SKIP] 实验 2 需要 --train-txt 和 --val-txt 参数")
        return

    tile_sizes = [int(s.strip()) for s in args.tile_sizes.split(",")]

    for tile_size in tile_sizes:
        size_dir = exp_dir / f"tile_{tile_size}"
        tiles_dir = size_dir / "tiles"

        # 步骤 1: 生成切片
        cmd_tiles = [
            sys.executable, str(SCRIPT_DIR / "prepare_tiles.py"),
            "--images-dir", args.images_dir,
            "--masks-dir", args.masks_dir,
            "--output-dir", str(tiles_dir),
            "--tile-size", str(tile_size),
            "--pad",
            "--split-txt", args.train_txt,
        ]
        run_command(cmd_tiles, f"实验 2: 生成 {tile_size}x{tile_size} 训练切片")

        # 生成验证集切片
        val_tiles_dir = size_dir / "val_tiles"
        cmd_val_tiles = [
            sys.executable, str(SCRIPT_DIR / "prepare_tiles.py"),
            "--images-dir", args.images_dir,
            "--masks-dir", args.masks_dir,
            "--output-dir", str(val_tiles_dir),
            "--tile-size", str(tile_size),
            "--pad",
            "--split-txt", args.val_txt,
        ]
        run_command(cmd_val_tiles, f"实验 2: 生成 {tile_size}x{tile_size} 验证切片")

        # 步骤 2: 训练
        train_cmd = [
            sys.executable, str(SCRIPT_DIR / "train_segmentation.py"),
            "--images-dir", str(tiles_dir / "images"),
            "--masks-dir", str(tiles_dir / "masks"),
            "--train-txt", str(tiles_dir / "all_tiles.txt"),
            "--val-txt", str(val_tiles_dir / "all_tiles.txt"),
            "--save-dir", str(size_dir / "runs"),
            "--project-name", f"tile_{tile_size}_ablation",
            "--height", str(tile_size),
            "--width", str(tile_size),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--encoder-name", args.encoder_name,
            "--device", args.device,
        ]
        if args.amp:
            train_cmd.append("--amp")
        run_command(train_cmd, f"实验 2: 训练 tile_size={tile_size}")

    save_experiment_summary(exp_dir, "2", {
        "description": "切片尺寸消融",
        "tile_sizes": tile_sizes,
        "epochs": args.epochs,
    })


# ---------------------------------------------------------------------------
# 实验 3: 切片策略消融 (训练 overlap + 推理 overlap)
# ---------------------------------------------------------------------------

def run_experiment_3(args: argparse.Namespace) -> None:
    """切片策略消融：训练 overlap + 推理 overlap"""
    exp_dir = Path(args.output_dir) / "exp3_overlap_ablation"
    os.makedirs(exp_dir, exist_ok=True)

    if not args.train_txt or not args.val_txt:
        print("[SKIP] 实验 3 需要 --train-txt 和 --val-txt 参数")
        return

    tile_size = 256
    overlaps = [float(s.strip()) for s in args.overlaps.split(",")]
    sahi_overlaps = [float(s.strip()) for s in args.sahi_overlaps.split(",")]

    # 步骤 1: 不同训练 overlap 的训练
    for overlap in overlaps:
        overlap_str = f"{overlap:.2f}".replace(".", "")
        overlap_dir = exp_dir / f"train_overlap_{overlap_str}"
        tiles_dir = overlap_dir / "tiles"

        cmd_tiles = [
            sys.executable, str(SCRIPT_DIR / "prepare_tiles.py"),
            "--images-dir", args.images_dir,
            "--masks-dir", args.masks_dir,
            "--output-dir", str(tiles_dir),
            "--tile-size", str(tile_size),
            "--overlap", str(overlap),
            "--pad",
            "--split-txt", args.train_txt,
        ]
        run_command(cmd_tiles, f"实验 3: 生成 overlap={overlap} 训练切片")

        val_tiles_dir = overlap_dir / "val_tiles"
        cmd_val_tiles = [
            sys.executable, str(SCRIPT_DIR / "prepare_tiles.py"),
            "--images-dir", args.images_dir,
            "--masks-dir", args.masks_dir,
            "--output-dir", str(val_tiles_dir),
            "--tile-size", str(tile_size),
            "--pad",
            "--split-txt", args.val_txt,
        ]
        run_command(cmd_val_tiles, f"实验 3: 生成验证切片")

        train_cmd = [
            sys.executable, str(SCRIPT_DIR / "train_segmentation.py"),
            "--images-dir", str(tiles_dir / "images"),
            "--masks-dir", str(tiles_dir / "masks"),
            "--train-txt", str(tiles_dir / "all_tiles.txt"),
            "--val-txt", str(val_tiles_dir / "all_tiles.txt"),
            "--save-dir", str(overlap_dir / "runs"),
            "--project-name", f"overlap_{overlap_str}",
            "--height", str(tile_size),
            "--width", str(tile_size),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--encoder-name", args.encoder_name,
            "--device", args.device,
        ]
        if args.amp:
            train_cmd.append("--amp")
        run_command(train_cmd, f"实验 3: 训练 overlap={overlap}")

    # 步骤 2: 不同推理 SAHI overlap 的评估（使用已有 256 模型）
    if args.checkpoint_256:
        for sahi_ov in sahi_overlaps:
            ov_str = f"{sahi_ov:.2f}".replace(".", "")
            infer_dir = exp_dir / f"sahi_overlap_{ov_str}"
            os.makedirs(infer_dir, exist_ok=True)

            cmd = [
                sys.executable, str(SCRIPT_DIR / "infer_pytorch.py"),
                "--checkpoint", args.checkpoint_256,
                "--input", args.images_dir,
                "--output-dir", str(infer_dir / "infer"),
                "--sahi",
                "--sahi-overlap", str(sahi_ov),
                "--gt-dir", args.masks_dir,
                "--metrics-output", str(infer_dir / "metrics.csv"),
                "--device", args.device,
            ]
            run_command(cmd, f"实验 3: SAHI overlap={sahi_ov} 推理评估")

    save_experiment_summary(exp_dir, "3", {
        "description": "切片策略消融",
        "train_overlaps": overlaps,
        "sahi_overlaps": sahi_overlaps,
    })


# ---------------------------------------------------------------------------
# 实验 4: K-fold 交叉验证 / 多种子
# ---------------------------------------------------------------------------

def run_experiment_4(args: argparse.Namespace) -> None:
    """验证集可靠性：K-fold 交叉验证 + 多种子"""
    exp_dir = Path(args.output_dir) / "exp4_kfold_multiseed"
    os.makedirs(exp_dir, exist_ok=True)

    k = args.kfold
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    tile_size = 256

    # 方案 A: K-fold 交叉验证
    kfold_dir = exp_dir / f"kfold_{k}"
    os.makedirs(kfold_dir, exist_ok=True)

    # 生成 K-fold 划分
    cmd_kfold = [
        sys.executable, str(SCRIPT_DIR / "prepare_kfold_splits.py"),
        "--images-dir", args.images_dir,
        "--masks-dir", args.masks_dir,
        "--output-dir", str(kfold_dir / "splits"),
        "--k", str(k),
        "--seed", "42",
    ]
    run_command(cmd_kfold, f"实验 4: 生成 {k}-fold 划分")

    # 对每个 fold 训练
    for fold_idx in range(1, k + 1):
        fold_split_dir = kfold_dir / "splits" / f"fold_{fold_idx}"
        fold_train_txt = fold_split_dir / "train.txt"
        fold_val_txt = fold_split_dir / "val.txt"

        if not fold_train_txt.is_file():
            print(f"[SKIP] fold_{fold_idx}/train.txt 不存在")
            continue

        # 生成切片
        fold_tiles_dir = kfold_dir / f"fold_{fold_idx}" / "train_tiles"
        cmd_tiles = [
            sys.executable, str(SCRIPT_DIR / "prepare_tiles.py"),
            "--images-dir", args.images_dir,
            "--masks-dir", args.masks_dir,
            "--output-dir", str(fold_tiles_dir),
            "--tile-size", str(tile_size),
            "--pad",
            "--split-txt", str(fold_train_txt),
        ]
        run_command(cmd_tiles, f"实验 4: fold {fold_idx} 生成训练切片")

        fold_val_tiles_dir = kfold_dir / f"fold_{fold_idx}" / "val_tiles"
        cmd_val_tiles = [
            sys.executable, str(SCRIPT_DIR / "prepare_tiles.py"),
            "--images-dir", args.images_dir,
            "--masks-dir", args.masks_dir,
            "--output-dir", str(fold_val_tiles_dir),
            "--tile-size", str(tile_size),
            "--pad",
            "--split-txt", str(fold_val_txt),
        ]
        run_command(cmd_val_tiles, f"实验 4: fold {fold_idx} 生成验证切片")

        # 训练
        train_cmd = [
            sys.executable, str(SCRIPT_DIR / "train_segmentation.py"),
            "--images-dir", str(fold_tiles_dir / "images"),
            "--masks-dir", str(fold_tiles_dir / "masks"),
            "--train-txt", str(fold_tiles_dir / "all_tiles.txt"),
            "--val-txt", str(fold_val_tiles_dir / "all_tiles.txt"),
            "--save-dir", str(kfold_dir / f"fold_{fold_idx}" / "runs"),
            "--project-name", f"kfold_fold{fold_idx}",
            "--height", str(tile_size),
            "--width", str(tile_size),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--encoder-name", args.encoder_name,
            "--device", args.device,
        ]
        if args.amp:
            train_cmd.append("--amp")
        run_command(train_cmd, f"实验 4: K-fold 训练 fold {fold_idx}/{k}")

    # 方案 B: 多种子划分
    for seed in seeds:
        seed_dir = exp_dir / f"seed_{seed}"
        os.makedirs(seed_dir, exist_ok=True)

        # 生成划分
        seed_split_dir = seed_dir / "splits"
        cmd_split = [
            sys.executable, str(SCRIPT_DIR / "prepare_splits.py"),
            "--images-dir", args.images_dir,
            "--masks-dir", args.masks_dir,
            "--output-dir", str(seed_split_dir),
            "--seed", str(seed),
        ]
        run_command(cmd_split, f"实验 4: 种子 {seed} 生成划分")

        seed_train_txt = seed_split_dir / "train.txt"
        seed_val_txt = seed_split_dir / "val.txt"

        if not seed_train_txt.is_file():
            print(f"[SKIP] seed_{seed}/train.txt 生成失败")
            continue

        # 生成切片
        seed_tiles_dir = seed_dir / "train_tiles"
        cmd_tiles = [
            sys.executable, str(SCRIPT_DIR / "prepare_tiles.py"),
            "--images-dir", args.images_dir,
            "--masks-dir", args.masks_dir,
            "--output-dir", str(seed_tiles_dir),
            "--tile-size", str(tile_size),
            "--pad",
            "--split-txt", str(seed_train_txt),
        ]
        run_command(cmd_tiles, f"实验 4: seed={seed} 生成训练切片")

        seed_val_tiles_dir = seed_dir / "val_tiles"
        cmd_val_tiles = [
            sys.executable, str(SCRIPT_DIR / "prepare_tiles.py"),
            "--images-dir", args.images_dir,
            "--masks-dir", args.masks_dir,
            "--output-dir", str(seed_val_tiles_dir),
            "--tile-size", str(tile_size),
            "--pad",
            "--split-txt", str(seed_val_txt),
        ]
        run_command(cmd_val_tiles, f"实验 4: seed={seed} 生成验证切片")

        # 训练
        train_cmd = [
            sys.executable, str(SCRIPT_DIR / "train_segmentation.py"),
            "--images-dir", str(seed_tiles_dir / "images"),
            "--masks-dir", str(seed_tiles_dir / "masks"),
            "--train-txt", str(seed_tiles_dir / "all_tiles.txt"),
            "--val-txt", str(seed_val_tiles_dir / "all_tiles.txt"),
            "--save-dir", str(seed_dir / "runs"),
            "--project-name", f"seed_{seed}",
            "--height", str(tile_size),
            "--width", str(tile_size),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--encoder-name", args.encoder_name,
            "--seed", str(seed),
            "--device", args.device,
        ]
        if args.amp:
            train_cmd.append("--amp")
        run_command(train_cmd, f"实验 4: 多种子训练 seed={seed}")

    save_experiment_summary(exp_dir, "4", {
        "description": "K-fold 交叉验证 + 多种子",
        "kfold": k,
        "seeds": seeds,
    })


# ---------------------------------------------------------------------------
# 实验 5: 原图训练公平基线
# ---------------------------------------------------------------------------

def run_experiment_5(args: argparse.Namespace) -> None:
    """原图训练公平基线：统一验证集"""
    exp_dir = Path(args.output_dir) / "exp5_fullimage_baseline"
    os.makedirs(exp_dir, exist_ok=True)

    if not args.train_txt or not args.val_txt:
        print("[SKIP] 实验 5 需要 --train-txt 和 --val-txt 参数")
        return

    baseline_sizes = [int(s.strip()) for s in args.baseline_sizes.split(",")]

    for size in baseline_sizes:
        size_dir = exp_dir / f"size_{size}"
        os.makedirs(size_dir, exist_ok=True)

        train_cmd = [
            sys.executable, str(SCRIPT_DIR / "train_segmentation.py"),
            "--images-dir", args.images_dir,
            "--masks-dir", args.masks_dir,
            "--train-txt", args.train_txt,
            "--val-txt", args.val_txt,
            "--save-dir", str(size_dir / "runs"),
            "--project-name", f"fullimage_{size}",
            "--height", str(size),
            "--width", str(size),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--encoder-name", args.encoder_name,
            "--device", args.device,
        ]
        if args.amp:
            train_cmd.append("--amp")
        run_command(train_cmd, f"实验 5: 原图训练 size={size}")

    save_experiment_summary(exp_dir, "5", {
        "description": "原图训练公平基线",
        "sizes": baseline_sizes,
    })


# ---------------------------------------------------------------------------
# 实验 6: 最终测试集评估
# ---------------------------------------------------------------------------

def run_experiment_6(args: argparse.Namespace) -> None:
    """最终测试集评估"""
    exp_dir = Path(args.output_dir) / "exp6_test_set_eval"
    os.makedirs(exp_dir, exist_ok=True)

    test_txt = args.test_txt
    if not test_txt:
        print("[SKIP] 实验 6 需要 --test-txt 参数")
        return

    checkpoints = []
    if args.checkpoint_256:
        checkpoints.append(("256_tile_model", args.checkpoint_256))
    if args.checkpoint_640:
        checkpoints.append(("640_orig_model", args.checkpoint_640))

    if not checkpoints:
        print("[SKIP] 实验 6 需要至少一个 checkpoint")
        return

    for model_name, checkpoint_path in checkpoints:
        model_dir = exp_dir / model_name
        os.makedirs(model_dir, exist_ok=True)

        # SAHI 推理评估
        cmd_sahi = [
            sys.executable, str(SCRIPT_DIR / "infer_pytorch.py"),
            "--checkpoint", checkpoint_path,
            "--input", args.images_dir,
            "--output-dir", str(model_dir / "sahi_infer"),
            "--sahi",
            "--sahi-overlap", "0.2",
            "--gt-dir", args.masks_dir,
            "--metrics-output", str(model_dir / "sahi_metrics.csv"),
            "--overlay",
            "--device", args.device,
        ]
        run_command(cmd_sahi, f"实验 6: 测试集 SAHI 评估 - {model_name}")

        # evaluate.py 固定尺寸评估
        cmd_eval = [
            sys.executable, str(SCRIPT_DIR / "evaluate.py"),
            "--checkpoint", checkpoint_path,
            "--images-dir", args.images_dir,
            "--masks-dir", args.masks_dir,
            "--val-txt", test_txt,
            "--output-dir", str(model_dir / "fixed_eval"),
            "--device", args.device,
        ]
        run_command(cmd_eval, f"实验 6: 测试集固定尺寸评估 - {model_name}")

        # evaluate.py 动态推理评估
        cmd_dynamic = [
            sys.executable, str(SCRIPT_DIR / "evaluate.py"),
            "--checkpoint", checkpoint_path,
            "--images-dir", args.images_dir,
            "--masks-dir", args.masks_dir,
            "--val-txt", test_txt,
            "--output-dir", str(model_dir / "dynamic_eval"),
            "--dynamic",
            "--device", args.device,
        ]
        run_command(cmd_dynamic, f"实验 6: 测试集动态推理评估 - {model_name}")

    save_experiment_summary(exp_dir, "6", {
        "description": "测试集最终评估",
        "models": [name for name, _ in checkpoints],
        "test_txt": str(test_txt),
    })


# ---------------------------------------------------------------------------
# 实验 7: 定性分析
# ---------------------------------------------------------------------------

def run_experiment_7(args: argparse.Namespace) -> None:
    """定性分析：可视化对比"""
    exp_dir = Path(args.output_dir) / "exp7_qualitative_analysis"
    os.makedirs(exp_dir, exist_ok=True)

    val_txt = args.val_txt or args.test_txt
    if not val_txt:
        print("[SKIP] 实验 7 需要 --val-txt 或 --test-txt 参数")
        return

    checkpoints = []
    if args.checkpoint_256:
        checkpoints.append(("256_tile_model", args.checkpoint_256))
    if args.checkpoint_640:
        checkpoints.append(("640_orig_model", args.checkpoint_640))

    if not checkpoints:
        print("[SKIP] 实验 7 需要至少一个 checkpoint")
        return

    for model_name, checkpoint_path in checkpoints:
        model_dir = exp_dir / model_name
        os.makedirs(model_dir, exist_ok=True)

        # SAHI 推理 + overlay 可视化
        cmd = [
            sys.executable, str(SCRIPT_DIR / "infer_pytorch.py"),
            "--checkpoint", checkpoint_path,
            "--input", args.images_dir,
            "--output-dir", str(model_dir / "visual"),
            "--sahi",
            "--sahi-overlap", "0.3",
            "--gt-dir", args.masks_dir,
            "--metrics-output", str(model_dir / "metrics.csv"),
            "--overlay",
            "--device", args.device,
        ]
        run_command(cmd, f"实验 7: 可视化推理 - {model_name}")

    save_experiment_summary(exp_dir, "7", {
        "description": "定性分析可视化",
        "models": [name for name, _ in checkpoints],
        "tip": "对比各模型在 visual/ 目录下的 overlay 图，关注边界区域、小目标和大目标的分割质量",
    })


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

EXPERIMENT_MAP = {
    "1": ("公平对比：SAHI 原图推理 + GT 对比", run_experiment_1),
    "2": ("消融实验：切片尺寸", run_experiment_2),
    "3": ("切片策略消融：训练 overlap + 推理 overlap", run_experiment_3),
    "4": ("验证集可靠性：K-fold + 多种子", run_experiment_4),
    "5": ("原图训练公平基线", run_experiment_5),
    "6": ("最终测试集评估", run_experiment_6),
    "7": ("定性分析", run_experiment_7),
}


def main() -> None:
    args = parse_args()
    ensure_split_files(args)

    if args.experiments.strip().lower() == "all":
        experiment_ids = sorted(EXPERIMENT_MAP.keys())
    else:
        experiment_ids = [s.strip() for s in args.experiments.split(",")]
        for eid in experiment_ids:
            if eid not in EXPERIMENT_MAP:
                raise ValueError(f"未知实验编号: {eid}，可选: {list(EXPERIMENT_MAP.keys())}")

    os.makedirs(args.output_dir, exist_ok=True)

    master_summary = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments_requested": experiment_ids,
        "args": vars(args),
    }

    print(f"\n{'#' * 70}")
    print(f"# 256 切片有效性验证实验")
    print(f"# 实验列表: {experiment_ids}")
    print(f"# 输出目录: {args.output_dir}")
    print(f"{'#' * 70}\n")

    for eid in experiment_ids:
        desc, func = EXPERIMENT_MAP[eid]
        print(f"\n{'*' * 70}")
        print(f"* 开始实验 {eid}: {desc}")
        print(f"{'*' * 70}\n")
        try:
            func(args)
        except Exception as e:
            print(f"[ERROR] 实验 {eid} 出错: {e}")
            master_summary[f"exp{eid}_error"] = str(e)

    master_summary["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary_path = Path(args.output_dir) / "master_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(master_summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'#' * 70}")
    print(f"# 实验完成")
    print(f"# 结果摘要: {summary_path}")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
