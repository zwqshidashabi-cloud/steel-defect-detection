"""
SE-YOLO: 钢材缺陷检测训练脚本

使用方式:
  cd D:\projects\steel_detection
  D:\Anaconda\python.exe scripts\train.py

实验组:
  Exp A: YOLOv8n baseline
  Exp B: SE-YOLO (本工作)
"""

import os, sys
sys.path.insert(0, r"D:\projects\steel_detection")
os.chdir(r"D:\projects\steel_detection")

import ultra_mod.register  # 注册自定义模块
from ultralytics import YOLO


def train_yolov8n_baseline():
    """Exp A: YOLOv8n baseline on NEU-DET"""
    print("=" * 60)
    print("Exp A: YOLOv8n baseline")
    print("=" * 60)

    model = YOLO("yolov8n.pt")  # 预训练
    results = model.train(
        data="data/NEU-DET/data.yaml",
        epochs=300,
        batch=32,
        imgsz=640,
        optimizer="AdamW",
        lr0=1e-3,
        weight_decay=5e-4,
        warmup_epochs=3,
        cos_lr=True,
        augment=True,
        mosaic=0.5,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        patience=50,
        device=0,
        project="runs/baseline",
        name="yolov8n-neu",
        exist_ok=True,
    )
    return results


def train_se_yolo():
    """Exp B: SE-YOLO on NEU-DET"""
    print("=" * 60)
    print("Exp B: SE-YOLO")
    print("=" * 60)

    model = YOLO("ultra_mod/cfg/models/se-yolo.yaml")
    results = model.train(
        data="data/NEU-DET/data.yaml",
        epochs=300,
        batch=32,
        imgsz=640,
        optimizer="AdamW",
        lr0=1e-3,
        weight_decay=5e-4,
        warmup_epochs=3,
        cos_lr=True,
        augment=True,
        mosaic=0.5,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        patience=50,
        device=0,
        project="runs/se-yolo",
        name="se-yolo-neu",
        exist_ok=True,
    )
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="只跑 baseline")
    parser.add_argument("--se", action="store_true", help="只跑 SE-YOLO")
    args = parser.parse_args()

    if not args.baseline and not args.se:
        args.baseline = args.se = True

    if args.baseline:
        train_yolov8n_baseline()
    if args.se:
        train_se_yolo()

    print("\n[Done] 训练完成!")
