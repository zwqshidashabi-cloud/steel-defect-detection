"""
test_model.py: 测试自定义模块注册 + 模型加载

使用方法:
    cd D:\projects\steel_detection
    python test_model.py
"""

import os
os.chdir(r"D:\projects\steel_detection")
import sys
sys.path.insert(0, r"D:\projects\steel_detection")

# 1. 注册自定义模块
import ultra_mod.register

# 2. 加载模型
from ultralytics import YOLO

print("[Test] 加载 SE-YOLO 模型配置...")
model = YOLO("ultra_mod/cfg/models/se-yolo.yaml", task="detect")

print(f"\n[OK] 模型加载成功!")
print(f"   层数: {len(model.model.model)}")
total_params = sum(p.numel() for p in model.parameters())
print(f"   参数量: {total_params:,}")

# 3. 验证前向传播 (随机输入)
import torch
x = torch.randn(1, 3, 640, 640)
model.model.eval()
with torch.no_grad():
    out = model.model(x)

print(f"\n[OK] 前向传播成功!")
if isinstance(out, list):
    for i, o in enumerate(out):
        print(f"   输出 {i}: {list(o.shape)}")

print(f"\n[PASS] 所有测试通过!")
print(f"现在可以用以下命令训练:")
print(f"   python scripts/train.py")
