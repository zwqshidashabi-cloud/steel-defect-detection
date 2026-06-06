"""
PGI_Detect: 带 P2 辅助监督头的检测头（Programmable Gradient Information）

【核心逻辑】
训练阶段:
  主分支(P3/P4/P5) + P2辅助分支 → 各自计算 Loss → 联合反向传播

推理/导出阶段:
  P2 辅助分支被完全阻断, 零额外算力

【与 Ultralytics Detect 的兼容性】
- 继承 nn.Module, 保持 forward 签名一致
- 输出格式与原始 Detect 对齐:
  训练: (loss, None) 或 loss dict
  推理: (B, num_anchors, 4+num_classes, H, W) 的 list
"""

import torch
import torch.nn as nn
import math


# ════════════════════════════════════════════════════════════
# 基础卷积组件（引用 gsconv.py 中的 Conv, 内联避免循环依赖）
# ════════════════════════════════════════════════════════════

def autopad(k, p=None, d=1):
    if d > 1:
        k = [d * (x - 1) + 1 for x in k] if isinstance(k, (list, tuple)) else d * (k - 1) + 1
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        p = autopad(k)
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DFL(nn.Module):
    """Distribution Focal Loss (DFL) 模块 — 与 Ultralytics 兼容"""

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1))\
            .view(b, 4, a)


# ════════════════════════════════════════════════════════════
# PGI_Detect: 带 P2 辅助分支的可编程梯度信息检测头
# ════════════════════════════════════════════════════════════

class PGI_Detect(nn.Module):
    """
    PGI 检测头 — 训练时双分支联合监督, 推理时零开销

    与 Ultralytics parse_model 兼容:
        yaml:  - [[P3, P4, P5, P2], 1, PGI_Detect, [nc]]
        call:  PGI_Detect(ch_list, nc)
        ch_list = [c3, c4, c5, c2]  (前 3 个为主检测头, 最后 1 个为 P2 辅助)

    Args:
        ch: list of 4 ints, 各检测层通道数 [P3, P4, P5, P2]
        nc: 类别数 (NEU-DET = 6)
    """

    def __init__(self, ch, nc=6):
        super().__init__()
        self.nc = nc
        self.no = nc + 64   # 每个 anchor: reg(64) + cls(nc)
        self.export = False
        self.training = True

        # 从 ch 推断层数: 前 num_layers 个为主分支, 最后 1 个为辅助
        self.num_layers = len(ch) - 1  # 通常 = 3 (P3/P4/P5)
        ch_main = ch[:self.num_layers]   # [c3, c4, c5]
        ch_aux = ch[self.num_layers]     # c2 (P2)

        self.stride = torch.zeros(self.num_layers)

        # ── 主检测头 (P3/P4/P5) ──
        self.main_layers = nn.ModuleList()
        for c_in in ch_main:
            self.main_layers.append(nn.Sequential(
                Conv(c_in, c_in, 3),
                nn.Conv2d(c_in, self.no, 1)
            ))

        # ── P2 辅助分支 ──
        self.aux_p2 = nn.Sequential(
            Conv(ch_aux, ch_aux, 3),
            nn.Conv2d(ch_aux, self.no, 1)
        )

        # ── DFL 模块 ──
        self.dfl = DFL(16)

    def forward(self, x):
        """
        Args:
            x: list of features [P3, P4, P5, P2]
               P3/P4/P5 是主检测头输入
               P2 是辅助分支输入（仅在训练时提供）

        Returns:
            训练: dict with 'main' and 'aux' predictions
            推理: list of main predictions (与原始 Detect 格式一致)
        """
        # 将输入拆分为主分支和辅助分支
        main_feats = x[:self.num_layers]   # [P3, P4, P5]
        aux_feat = x[self.num_layers]      # P2

        # ── 主分支推理 ──
        main_out = []
        for i, feat in enumerate(main_feats):
            main_out.append(self.main_layers[i](feat))

        # ── 推理/导出模式: 只返回主分支 ──
        if not self.training or self.export:
            return main_out

        # ── 训练模式: 额外计算 P2 辅助分支 ──
        aux_out = self.aux_p2(aux_feat)

        return {
            "main": main_out,       # list of 3 tensors (P3/P4/P5)
            "aux": [aux_out],       # list of 1 tensor (P2)
        }

    def bias_init(self):
        """初始化检测头的偏置 (与 Ultralytics Detect 一致)"""
        # 对每个检测层初始化
        for m in self.main_layers:
            m[-1].bias.data[:64] = 0.0           # reg 分支
            m[-1].bias.data[64:] = -math.log((1 - 0.01) / 0.01)  # cls 分支

        # 辅助分支
        self.aux_p2[-1].bias.data[:64] = 0.0
        self.aux_p2[-1].bias.data[64:] = -math.log((1 - 0.01) / 0.01)

    def switch_to_export(self):
        """切换到导出模式: 训练时调用 export=True 后自动剥离 P2 分支"""
        self.export = True


# ════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PGI_Detect 维度测试")
    print("=" * 60)

    B = 2
    nc = 6          # NEU-DET 类别数
    H, W = 80, 80   # P3 层尺寸

    # 模拟输入: P3(80x80), P4(40x40), P5(20x20), P2(160x160)
    feats = [
        torch.randn(B, 64, 80, 80),    # P3
        torch.randn(B, 128, 40, 40),   # P4
        torch.randn(B, 256, 20, 20),   # P5
        torch.randn(B, 32, 160, 160),  # P2 (aux)
    ]
    ch = [64, 128, 256, 32]

    # ── 测试 1: 推理模式 ──
    print("\n[Test 1] 推理模式 (training=False)")
    model = PGI_Detect(ch=ch, nc=nc)
    model.eval()
    out = model(feats)
    assert isinstance(out, list) and len(out) == 3, "推理模式应输出 list[3]"
    for i, o in enumerate(out):
        expected_c = model.no  # = nc + 64 (reg=64, cls=nc)
        print(f"  主分支 P{i+3}: {list(o.shape)}  ← (B={B}, {expected_c}, H={H//(2**i)}, W={W//(2**i)})")
        assert o.shape[1] == expected_c, f"P{i+3} 通道数应为 {expected_c}"
    print("  [OK]")

    # ── 测试 2: 训练模式 ──
    print("\n[Test 2] 训练模式 (training=True)")
    model.train()
    out_train = model(feats)
    assert isinstance(out_train, dict), "训练模式应输出 dict"
    assert "main" in out_train and "aux" in out_train, "应包含 main 和 aux"
    assert len(out_train["main"]) == 3, "main 应有 3 层"
    assert len(out_train["aux"]) == 1, "aux 应有 1 层 (P2)"
    print(f"  main 分支: {[list(o.shape) for o in out_train['main']]}")
    print(f"  aux P2 分支: {list(out_train['aux'][0].shape)}")
    assert out_train["aux"][0].shape[2] == 160, "P2 应保持 160x160"
    print("  [OK]")

    # ── 测试 3: 导出模式 (P2 剥离) ──
    print("\n[Test 3] 导出模式 (export=True)")
    model.eval()
    model.switch_to_export()
    out_export = model(feats)
    assert isinstance(out_export, list) and len(out_export) == 3, "导出时应只输出主分支"
    print(f"  导出输出维度: {[list(o.shape) for o in out_export]}")
    print("  [OK] 导出时 P2 已剥离, 零额外算力")

    # ── 测试 4: bias_init ──
    print("\n[Test 4] bias_init")
    model.bias_init()
    print("  [OK] bias 初始化完成")

    # ── 测试 5: 梯度回传 ──
    print("\n[Test 5] 梯度回传")
    model.train()
    model.export = False  # 清除 export 标志, 恢复训练模式
    feats_grad = [f.requires_grad_(True) for f in feats]
    out_grad = model(feats_grad)
    loss = sum(o.sum() for o in out_grad["main"]) + out_grad["aux"][0].sum()
    loss.backward()
    all_grad = all(f.grad is not None and f.grad.abs().sum().item() > 0 for f in feats_grad)
    assert all_grad, "部分特征图无梯度"
    print(f"  total loss = {loss.item():.2f}")
    print(f"  所有特征图梯度正常: {[f.grad.abs().sum().item() for f in feats_grad]}")
    print("  [OK] 梯度回传通过")

    # ── 参数统计 ──
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[PGI_Detect 参数统计]:")
    print(f"  总参数量:     {total_params:,}")
    print(f"  可训练参数:   {trainable_params:,}")
    # 辅助分支占比
    aux_params = sum(p.numel() for p in model.aux_p2.parameters())
    print(f"  辅助分支参数量: {aux_params:,} ({100*aux_params/total_params:.1f}%)")
    print(f"  推理时辅助分支剥离: 零额外算力")

    print(f"\n{'=' * 60}")
    print("[PASS] PGI_Detect 全部测试通过!")
    print(f"{'=' * 60}")
