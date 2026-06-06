"""
GSConv + GSC_Bottleneck_Cross + C2f_GSC_Cross: 颈部网络的算力降载与特征融合

【特征接力联动逻辑】
SCA-Block 截留的 SPD 高频特征 -> 通道极其密集 ->
GSConv 的 Channel Shuffle 进行低 FLOPs 特征洗牌 ->
GSC_Bottleneck_Cross 的残差直连保护龟裂边缘特征不被打碎

【兼容性】
初始化参数与 Ultralytics C2f 完全对齐: c1, c2, n, shortcut, g, e
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════
# 基础卷积组件（与 Ultralytics 风格对齐）
# ════════════════════════════════════════════════════════════

def autopad(k, p=None, d=1):
    """与 Ultralytics 一致的自动 padding 计算，支持 tuple kernel"""
    if d > 1:
        if isinstance(k, int):
            k = d * (k - 1) + 1
        else:
            k = [d * (x - 1) + 1 for x in k]
    if p is None:
        if isinstance(k, int):
            p = k // 2
        else:
            p = [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """标准卷积: Conv2d + BN + SiLU，与 Ultralytics Conv 兼容"""

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


class DWConv(Conv):
    """深度可分离卷积: groups = c1 (深度卷积)"""

    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__(c1, c2, k, s, g=min(c1, c2), act=act)


# ════════════════════════════════════════════════════════════
# GSConv: 标准卷积 -> DWConv -> Concat -> Channel Shuffle
# ════════════════════════════════════════════════════════════

class GSConv(nn.Module):
    """
    GSConv - 轻量化卷积
    流程:   Conv1x1 -> DWConv3x3 -> Concat -> Channel Shuffle
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2  # c2 必须为偶数
        self.cv1 = Conv(c1, c_, k, s, g, act)
        self.cv2 = DWConv(c_, c_, 3, 1, act)

    def forward(self, x):
        x1 = self.cv1(x)                         # (B, c_, H, W)
        x2 = self.cv2(x1)                        # (B, c_, H, W)

        out = torch.cat([x1, x2], dim=1)          # (B, 2c_, H, W)

        # Channel Shuffle: 交错排列通道
        b, c, h, w = out.shape
        out = out.reshape(b, 2, c // 2, h, w)
        out = out.transpose(1, 2).reshape(b, c, h, w)

        return out


# ════════════════════════════════════════════════════════════
# GSC_Bottleneck_Cross: GSConv + 残差特征重用
# ════════════════════════════════════════════════════════════

class GSC_Bottleneck_Cross(nn.Module):
    """
    基于 GSConv 的 Bottleneck，强制引入跨层残差连接 (Cross)

    结构:
        input -> Conv1x1(reduce) -> GSConv3x3 -> Conv3x3 -> output
          |                                                   ^
          +---------------- shortcut (if c1==c2) -------------+

    【物理意义】
    Channel Shuffle 可能打碎钢材裂纹微弱边缘特征。
    残差直连确保高频细节无损绕过 GSConv，实现算力降载与特征保护平衡。
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, k[0], 1)           # 1x1 降维
        self.gsc = GSConv(c_, c_, 3, 1, g)         # GSConv 混合
        self.cv2 = Conv(c_, c2, k[1], 1, g)        # 3x3 深度空间精炼

        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        x = self.cv1(x)      # c1 -> c_
        x = self.gsc(x)      # c_ -> c_ (含 shuffle)
        x = self.cv2(x)      # c_ -> c2
        if self.add:
            x = x + identity
        return x


# ════════════════════════════════════════════════════════════
# C2f_GSC_Cross: YOLOv8 C2f 兼容结构，内部替换为 GSC_Bottleneck_Cross
# ════════════════════════════════════════════════════════════

class C2f_GSC_Cross(nn.Module):
    """
    C2f_GSC_Cross - 轻量级特征融合模块

    与 YOLOv8 原版 C2f 完全兼容:
      - 参数签名: c1, c2, n, shortcut, g, e
      - 内部逻辑: Split -> n x Bottleneck (收集全部输出) -> Concat -> Conv1x1
      - 可直接替换 yaml 中的 'C2f' 为 'C2f_GSC_Cross'
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)               # c1 -> 2c
        self.cv2 = Conv((2 + n) * self.c, c2, 1)             # (2+n)c -> c2

        self.m = nn.ModuleList(
            GSC_Bottleneck_Cross(self.c, self.c, shortcut, g,
                                  k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        """前向传播 — 与 YOLOv8 C2f 完全一致的行为"""
        # cv1 降维 + chunk: y = [x_left(c), x_right(c)]
        y = list(self.cv1(x).chunk(2, dim=1))

        # 逐级通过 bottleneck，每次用 y[-1] 作为输入
        # 这样每步中间结果都会被保留用于最后的 concat
        y.extend(m(y[-1]) for m in self.m)

        # 拼接所有: (2+n) 个 tensor 各 c 通道 = (2+n)*c
        out = torch.cat(y, dim=1)

        # cv2 融合: (B, c2, H, W)
        out = self.cv2(out)
        return out


# ════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("GSConv / GSC_Bottleneck_Cross / C2f_GSC_Cross 测试")
    print("=" * 60)

    B, H, W = 2, 64, 64

    # 测试 1: GSConv
    print("\n[Test 1] GSConv")
    x1 = torch.randn(B, 32, H, W)
    m1 = GSConv(32, 64)
    o1 = m1(x1)
    print(f"  Input:  {list(x1.shape)}")
    print(f"  Output: {list(o1.shape)}")
    assert o1.shape == (B, 64, H, W)
    print("  [OK]")

    # 测试 2: GSC_Bottleneck_Cross (no shortcut, c1!=c2)
    print("\n[Test 2] GSC_Bottleneck_Cross (shortcut=False)")
    x2 = torch.randn(B, 32, H, W)
    m2 = GSC_Bottleneck_Cross(32, 64, shortcut=False, e=1.0)
    o2 = m2(x2)
    assert o2.shape == (B, 64, H, W)
    print("  [OK]")

    # 测试 3: GSC_Bottleneck_Cross (shortcut, c1==c2)
    print("\n[Test 3] GSC_Bottleneck_Cross (shortcut=True, c1=c2)")
    x3 = torch.randn(B, 64, H, W)
    m3 = GSC_Bottleneck_Cross(64, 64, shortcut=True, e=1.0)
    o3 = m3(x3)
    assert o3.shape == (B, 64, H, W) and m3.add == True
    print("  [OK]")

    # 测试 4: C2f_GSC_Cross n=1
    print("\n[Test 4] C2f_GSC_Cross (n=1, e=0.5)")
    x4 = torch.randn(B, 32, H, W)
    m4 = C2f_GSC_Cross(32, 64, n=1, e=0.5)
    o4 = m4(x4)
    # c=32, cv1:32->64, split->各32, GSC(32->32), concat(32+32=64), cv2:64->64
    assert o4.shape == (B, 64, H, W)
    print("  [OK]")

    # 测试 5: C2f_GSC_Cross n=3
    print("\n[Test 5] C2f_GSC_Cross (n=3, e=0.5)")
    x5 = torch.randn(B, 32, H, W)
    m5 = C2f_GSC_Cross(32, 64, n=3, e=0.5)
    o5 = m5(x5)
    assert o5.shape == (B, 64, H, W)
    print("  [OK]")

    # 测试 6: c1==c2 shortcut
    print("\n[Test 6] C2f_GSC_Cross (c1=c2, shortcut=True)")
    x6 = torch.randn(B, 64, H, W)
    m6 = C2f_GSC_Cross(64, 64, n=2, shortcut=True, e=0.5)
    o6 = m6(x6)
    assert o6.shape == (B, 64, H, W)
    print("  [OK]")

    # 测试 7: 梯度回传
    print("\n[Test 7] Gradient check")
    x7 = torch.randn(B, 32, H, W, requires_grad=True)
    m7 = C2f_GSC_Cross(32, 64, n=2)
    o7 = m7(x7)
    o7.sum().backward()
    assert x7.grad is not None and x7.grad.abs().sum().item() > 0
    print(f"  grad norm = {x7.grad.abs().sum().item():.6f}")
    print("  [OK]")

    # 参数统计
    print("\n[Params]:")
    for name, model in [
        ("GSConv", m1),
        ("GSC_Bottleneck_Cross", m3),
        ("C2f_GSC_Cross (n=1)", m4),
        ("C2f_GSC_Cross (n=3)", m5),
    ]:
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name:30s}: {params:>8,} params")

    print(f"\n{'=' * 60}")
    print("[PASS] All tests passed!")
    print(f"{'=' * 60}")
