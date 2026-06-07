"""
register.py: 将自定义模块注册到 Ultralytics 中
通过 monkey-patch 系统, 无需修改 site-packages 源码

用法:
    import register
    from ultralytics import YOLO
    model = YOLO("ultra-mod/cfg/models/se-yolo.yaml")
"""

import sys
import torch
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as mods

# ── 导入自定义模块 ──
sys.path.insert(0, r"D:\projects\steel_detection")

from ultra_mod.nn.modules.spd_conv import SCABlock
from ultra_mod.nn.modules.gsconv import C2f_GSC_Cross, GSConv, GSC_Bottleneck_Cross
from ultra_mod.nn.modules.head import PGI_Detect

# ── 注入到 ultralytics.nn.modules ──
mods.SCABlock = SCABlock
mods.C2f_GSC_Cross = C2f_GSC_Cross
mods.PGI_Detect = PGI_Detect

# ── Patch 1: 将自定义模块注入 tasks 命名空间 ──
# 这样 parse_model 中的 globals()[m] 可以找到它们
tasks.SCABlock = SCABlock
tasks.C2f_GSC_Cross = C2f_GSC_Cross
tasks.PGI_Detect = PGI_Detect

# ── Patch 2: 替换 parse_model ──
original_parse_model = tasks.parse_model

def patched_parse_model(d, ch, verbose=True):
    """包装原 parse_model, 注入自定义模块支持"""
    import ast
    from ultralytics.nn.modules import (
        AIFI, C1, C2, C2PSA, C3, C3TR, ELAN1, OBB, OBB26, PSA,
        SPP, SPPELAN, SPPF, A2C2f, AConv, ADown, Bottleneck,
        BottleneckCSP, C2f, C2fAttn, C2fCIB, C2fPSA, C3Ghost,
        C3k2, C3x, CBFuse, CBLinear, Classify, Concat, Conv,
        ConvTranspose, Detect, DWConv, DWConvTranspose2d, Focus,
        GhostBottleneck, GhostConv, HGBlock, HGStem, ImagePoolingAttn,
        Index, Pose, Pose26, RepC3, RepNCSPELAN4, ResNetLayer,
        RTDETRDecoder, Segment, Segment26, SCDown, SemanticSegment,
        TorchVision, v10Detect, WorldDetect, YOLOEDetect, YOLOESegment,
        YOLOESegment26, C2PSA, C2fAttn,
    )

    # ── 解析 yaml 配置 ──
    # 先解析所有变量, 再创建 scope
    legacy = True
    max_channels = float("inf")
    nc, act = d.get("nc"), d.get("activation")
    reg_max = d.get("reg_max", 16)
    end2end = d.get("end2end")
    depth, width = d.get("depth_multiple", 1.0), d.get("width_multiple", 1.0)
    scales = d.get("scales")
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
        depth, width, max_channels = scales[scale]

    # ⚠️ 现在创建 scope! nc/reg_max/end2end 等变量必须在 scope 中
    scope = locals()
    # 手动注入自定义模块和常用模块
    scope["SCABlock"] = SCABlock
    scope["C2f_GSC_Cross"] = C2f_GSC_Cross
    scope["PGI_Detect"] = PGI_Detect
    scope["Detect"] = Detect
    scope["nn"] = torch.nn

    # ── 构建模型 ──
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]

    base = frozenset({
        Classify, Conv, ConvTranspose, GhostConv, Bottleneck,
        GhostBottleneck, SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus,
        BottleneckCSP, C1, C2, C2f, C3k2, RepNCSPELAN4, ELAN1,
        ADown, AConv, SPPELAN, C2fAttn, C3, C3TR, C3Ghost,
        SCABlock, C2f_GSC_Cross,
        torch.nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3,
        PSA, SCDown, C2fCIB, A2C2f,
    })

    repeat_modules = frozenset({
        BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR,
        C3Ghost, C3x, RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f,
        C2f_GSC_Cross,
    })

    for i, (f, n, m_str, args) in enumerate(d["backbone"] + d["head"]):
        # 解析模块
        if "nn." in m_str:
            m = getattr(torch.nn, m_str[3:])
        elif "torchvision.ops." in m_str:
            m = getattr(__import__("torchvision").ops, m_str[16:])
        else:
            m = scope[m_str]  # 既能在 scope 找也能在 globals 找

        for j, a in enumerate(args):
            if isinstance(a, str):
                # 先查 scope (变量名如 "nc"), 再尝试 Python 字面量解析
                resolved = scope.get(a) if a in scope else None
                if resolved is None:
                    try:
                        resolved = ast.literal_eval(a)
                    except (ValueError, SyntaxError):
                        resolved = a  # 保持原字符串 (如 "nearest")
                args[j] = resolved

        n = n_ = max(round(n * depth), 1) if n > 1 else n

        if m in base:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = min(c2, max_channels) if isinstance(width, float) else c2
                c2 = int(min(c2, max_channels) * width / 8 + 0.5) * 8
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
        elif m in frozenset({Detect, PGI_Detect, WorldDetect, YOLOEDetect, Segment, Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26}):
            # Detect 类: 附加 reg_max, end2end, ch_list
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            if m is PGI_Detect:
                # PGI_Detect 签名: (ch, nc) → 从 args 中提取
                # detect 类附加的 args[-1] 是 ch list
                pass
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        else:
            c2 = ch[f] if isinstance(f, int) else ch[f[-1]]

        # 构建模块
        if m is PGI_Detect:
            # PGI_Detect 签名: PGI_Detect(nc, reg_max, end2end, ch) 与 Detect 一致
            ch_list = [ch[x] for x in f]
            nc_val = args[0] if isinstance(args[0], int) else 6
            if verbose:
                print(f"    PGI_Detect: ch_list={ch_list}, nc={nc_val}, reg_max={reg_max}")
            m_ = m(nc=nc_val, reg_max=reg_max, end2end=False, ch=ch_list)
        elif n > 1:
            m_ = torch.nn.Sequential(*(m(*args) for _ in range(n)))
        else:
            m_ = m(*args)

        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)

        if i == 0:
            ch = []
        ch.append(c2)

    return torch.nn.Sequential(*layers), sorted(save)

# ── 替换 parse_model ──
tasks.parse_model = patched_parse_model

if __name__ == "__main__":
    # 验证注册成功
    print(f"[Register] 自定义模块已注册到 Ultralytics {ultralytics.__version__}")
    print(f"  - SCABlock: {SCABlock}")
    print(f"  - C2f_GSC_Cross: {C2f_GSC_Cross}")
    print(f"  - PGI_Detect: {PGI_Detect}")

    # 尝试加载模型配置
    from ultralytics import YOLO
    import os
    os.chdir(r"D:\projects\steel_detection")

    try:
        model = YOLO("ultra-mod/cfg/models/se-yolo.yaml", task="detect")
        print(f"\n✅ 模型加载成功!")
        print(f"   层数: {len(model.model.model)}")
        total = sum(p.numel() for p in model.parameters())
        print(f"   参数量: {total:,}")
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
