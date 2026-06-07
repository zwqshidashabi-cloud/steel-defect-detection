"""
NEU-DET 钢材缺陷数据集下载与准备
自动从 kagglehub 下载 (无需 API key) 并转换为 YOLO 格式

使用方法:
    D:\Anaconda\python.exe scripts/download_neu_det.py
"""

import os, sys, xml.etree.ElementTree as ET, shutil
from pathlib import Path

DATA_DIR = Path(r"D:\projects\steel_detection\data\NEU-DET")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface",
           "rolled-in_scale", "scratches"]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASSES)}


def xml_to_yolo(xml_path):
    """VOC XML -> YOLO txt lines"""
    root = ET.parse(xml_path).getroot()
    w, h = int(root.find("size/width").text), int(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_TO_ID:
            continue
        bb = obj.find("bndbox")
        xmin, ymin = float(bb.find("xmin").text), float(bb.find("ymin").text)
        xmax, ymax = float(bb.find("xmax").text), float(bb.find("ymax").text)
        cx = ((xmin + xmax) / 2) / w
        cy = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{CLASS_TO_ID[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def process(src_img_dir, src_ann_dir, dst_img_dir, dst_lbl_dir):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    for d in src_img_dir.iterdir():
        if d.is_dir():
            for f in d.glob("*.jpg"):
                if not (dst_img_dir / f.name).exists():
                    shutil.copy2(str(f), str(dst_img_dir))
    for f in src_ann_dir.glob("*.xml"):
        lines = xml_to_yolo(f)
        if lines:
            (dst_lbl_dir / (f.stem + ".txt")).write_text("\n".join(lines))
    return len(list(dst_img_dir.glob("*.jpg"))), len(list(dst_lbl_dir.glob("*.txt")))


def main():
    # 检查已有数据
    if (DATA_DIR / "images" / "train").exists() and len(list((DATA_DIR / "images" / "train").glob("*.jpg"))) > 0:
        print(f"[OK] 数据集已存在")
        return

    # 下载
    print("[Download] 从 Kaggle 下载 NEU-DET...")
    try:
        import kagglehub
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "kagglehub", "-q"])
        import kagglehub
    src = Path(kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")) / "NEU-DET"

    # 转换 train
    n_img, n_lbl = process(src / "train" / "images", src / "train" / "annotations",
                            DATA_DIR / "images" / "train", DATA_DIR / "labels" / "train")
    print(f"  train: {n_img} imgs, {n_lbl} labels")

    # 转换 valid
    val_k = "validation" if (src / "validation").exists() else "valid"
    n_img, n_lbl = process(src / val_k / "images", src / val_k / "annotations",
                            DATA_DIR / "images" / "val", DATA_DIR / "labels" / "val")
    print(f"  val: {n_img} imgs, {n_lbl} labels")

    # data.yaml
    yaml = f"path: {str(DATA_DIR).replace(chr(92), '/')}\ntrain: images/train\nval: images/val\nnc: 6\nnames: {CLASSES}\n"
    (DATA_DIR / "data.yaml").write_text(yaml)
    print(f"[OK] NEU-DET 已就绪!")


if __name__ == "__main__":
    main()
