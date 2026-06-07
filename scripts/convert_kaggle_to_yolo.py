"""
将 kagglehub 下载的 NEU-DET 转换为 YOLO 格式

使用方法:
    D:\Anaconda\python.exe scripts/convert_kaggle_to_yolo.py
"""

import kagglehub
import os, sys, shutil, xml.etree.ElementTree as ET
from pathlib import Path

DATA_DIR = Path(r"D:\projects\steel_detection\data\NEU-DET")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface",
           "rolled-in_scale", "scratches"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def convert_xml_to_yolo(xml_path):
    """VOC XML -> YOLO format (cx, cy, w, h) normalized"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_TO_ID:
            continue
        cls_id = CLASS_TO_ID[name]
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def process_split(src_images, src_annotations, dst_img_dir, dst_lbl_dir):
    """处理一个 split (train/valid)"""
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    # 从类别子目录复制图片
    for class_dir in src_images.iterdir():
        if class_dir.is_dir():
            for img_path in class_dir.glob("*.jpg"):
                dst = dst_img_dir / img_path.name
                if not dst.exists():
                    shutil.copy2(str(img_path), str(dst))

    # 转换 XML 标注
    for xml_path in src_annotations.glob("*.xml"):
        lines = convert_xml_to_yolo(xml_path)
        if lines:
            (dst_lbl_dir / (xml_path.stem + ".txt")).write_text(
                "\n".join(lines), encoding="utf-8")

    img_count = len(list(dst_img_dir.glob("*.jpg")))
    lbl_count = len(list(dst_lbl_dir.glob("*.txt")))
    print(f"  [OK] {dst_img_dir.parent.name}: {img_count} imgs, {lbl_count} labels")
    return img_count, lbl_count


def main():
    # 下载路径
    kaggle_path = Path(kagglehub.dataset_download(
        "kaustubhdikshit/neu-surface-defect-database", force_download=False))

    dataset_root = kaggle_path / "NEU-DET"
    print(f"[Info] Kaggle 数据集: {dataset_root}")

    # 处理 train
    src_train = dataset_root / "train"
    process_split(
        src_train / "images",
        src_train / "annotations",
        DATA_DIR / "images" / "train",
        DATA_DIR / "labels" / "train",
    )

    # 处理 validation
    src_val = dataset_root / "validation" if (dataset_root / "validation").exists() else dataset_root / "valid"
    process_split(
        src_val / "images",
        src_val / "annotations",
        DATA_DIR / "images" / "val",
        DATA_DIR / "labels" / "val",
    )

    # 生成 data.yaml
    yaml_content = f"""
path: {str(DATA_DIR).replace(chr(92), '/')}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    (DATA_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"[OK] data.yaml 已生成")

    # 统计
    print(f"\n{'='*50}")
    print(f"NEU-DET 数据集准备完成!")
    print(f"{'='*50}")
    for split in ["train", "val"]:
        imgs = len(list((DATA_DIR / "images" / split).glob("*.jpg")))
        lbls = len(list((DATA_DIR / "labels" / split).glob("*.txt")))
        print(f"  {split}: {imgs} imgs, {lbls} labels")
    print(f"  类别: {CLASSES}")
    print(f"  位置: {DATA_DIR}")
    print(f"\n训练命令:")
    print(f"  D:\\Anaconda\\python.exe scripts/train.py --baseline")
    print(f"  D:\\Anaconda\\python.exe scripts/train.py --se")


if __name__ == "__main__":
    main()
