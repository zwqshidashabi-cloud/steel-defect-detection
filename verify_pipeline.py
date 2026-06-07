"""
验证数据管道和训练入口
直接运行: python verify_pipeline.py
"""

import sys, os
os.chdir(r"D:\projects\steel_detection")
sys.path.insert(0, r"D:\projects\steel_detection")

# 1. 注册自定义模块
import ultra_mod.register
print("[1/4] 自定义模块注册完成")

# 2. 加载 SE-YOLO 配置
from ultralytics import YOLO
model = YOLO("ultra_mod/cfg/models/se-yolo.yaml", task="detect")
total = sum(p.numel() for p in model.parameters())
print(f"[2/4] SE-YOLO 加载成功: {total:,} params")

# 3. 前向传播测试
import torch
x = torch.randn(1, 3, 640, 640)
model.model.eval()
with torch.no_grad():
    out = model.model(x)
print(f"[3/4] 前向传播成功: {len(out)} outputs")

# 4. 数据集验证
import yaml, glob
from PIL import Image
with open("data/NEU-DET/data.yaml") as f:
    d = yaml.safe_load(f)
imgs = glob.glob("data/NEU-DET/images/train/*.jpg")
lbls = glob.glob("data/NEU-DET/labels/train/*.txt")
print(f"[4/4] 数据集: {len(imgs)} imgs, {len(lbls)} labels")
print(f"      类别: {d['names']}")

print("\n" + "="*50)
print(" 全部通过! 可以开始训练")
print("="*50)
print("\n训练命令:")
print("  # Baseline (YOLOv8n):")
print("  python scripts/train.py --baseline")
print("\n  # SE-YOLO (本工作):")
print("  python scripts/train.py --se")
