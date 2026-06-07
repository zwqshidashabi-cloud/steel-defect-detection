"""
NEU-DET 钢材缺陷数据集下载与准备

数据集来源: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
"""

import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, r"D:\projects\steel_detection")

DATA_DIR = Path(r"D:\projects\steel_detection\data\NEU-DET")
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"

# 6类缺陷
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def create_dataset_yaml():
    """生成 data.yaml"""
    yaml_content = f"""
# NEU-DET 钢材缺陷检测数据集
path: {str(DATA_DIR).replace(chr(92), '/')}
train: images/train
val: images/val

nc: 6
names: {CLASSES}

# 数据增强
mosaic: 0.5
mixup: 0.1
copy_paste: 0.0
flipud: 0.5
fliplr: 0.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 10.0
translate: 0.1
scale: 0.5
shear: 2.0
"""
    yaml_path = DATA_DIR / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"[NEU-DET] data.yaml 已生成: {yaml_path}")


def prepare_dataset():
    """
    NEU-DET 数据集准备

    两种方式:
    1. 自动从 GitHub 源下载 (推荐)
    2. 手动下载后放在 data/NEU-DET/raw/
    """
    # ── 方式 1: 从 mirror 下载 ──
    os.makedirs(DATA_DIR, exist_ok=True)

    # NEU-DET 可从公开源下载
    # 使用 GitHub 上的转换版本
    url = (
        "https://github.com/kaustubhikd/NEU-DET/archive/refs/heads/main.zip"
    )

    zip_path = DATA_DIR / "neu_det_raw.zip"
    extract_path = DATA_DIR / "raw"

    if not (IMAGES_DIR / "train").exists():
        print(f"[NEU-DET] 下载数据集中...")
        print(f"  下载地址: {url}")
        print(f"  请手动下载或从 Kaggle 获取: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database")
        print(f"\n  下载后将文件放到 {extract_path} 目录下")
        print(f"  然后运行: python scripts/prepare_neu_det.py")

        # 创建目录结构
        os.makedirs(IMAGES_DIR / "train", exist_ok=True)
        os.makedirs(IMAGES_DIR / "val", exist_ok=True)
        os.makedirs(LABELS_DIR / "train", exist_ok=True)
        os.makedirs(LABELS_DIR / "val", exist_ok=True)

        print(f"\n[NEU-DET] 目录结构已创建:")
        print(f"  {DATA_DIR}/")
        print(f"    ├── images/train/   (训练图片)")
        print(f"    ├── images/val/     (验证图片)")
        print(f"    ├── labels/train/   (训练标签)")
        print(f"    ├── labels/val/     (验证标签)")
        print(f"    └── data.yaml       (数据集配置文件)")
    else:
        print(f"[NEU-DET] 数据集已存在: {IMAGES_DIR}")

    create_dataset_yaml()


if __name__ == "__main__":
    prepare_dataset()
