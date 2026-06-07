"""
SCA-Block (SPD-CA Block): 将 Coordinate Attention 深度融合到 SPD 折叠中的下采样模块

【创新点】
传统做法：SPD 先做像素折叠 (C→4C)，再接入独立的 CA 模块，两者串行。
本模块：将 CA 的 1D 全局池化操作在原始分辨率上计算后，进行相同的 SPD 折叠，
        使注意力权重与原特征在折叠后的空间位置上一一对应，实现强方位感知的无损下采样。

【特征接力逻辑】
SPD 物理截留极小目标像素 → CA 在无损的高频空间特征上聚焦长条形纹理方向
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCABlock(nn.Module):
    """
    SCA-Block: SPD + Coordinate Attention 深度融合下采样模块

    输入:  (B, C_in, H, W)    其中 H, W 均为偶数
    输出:  (B, C_out, H/2, W/2)

    Args:
        in_channels:  输入通道数 C_in
        out_channels: 输出通道数 C_out
        reduction:    CA 通道压缩倍率 (default: 16)
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16):
        super().__init__()

        r = max(1, in_channels // reduction)  # 确保压缩通道数至少为 1

        # ── CA 分支：在原始分辨率上计算方向注意力 ──
        # 共享 1×1 Conv 对 H-pool 和 W-pool 降维
        self.conv_ca_reduce = nn.Conv2d(in_channels, r, 1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(r)

        # SPD 折叠后注意力扩张到 in_channels
        # H 方向：(r*2 → in_channels)；每行有 2 个子像素偏移
        self.conv_expand_h = nn.Conv2d(r * 2, in_channels, 1, bias=False)
        # W 方向：(r*2 → in_channels)；每列有 2 个子像素偏移
        self.conv_expand_w = nn.Conv2d(r * 2, in_channels, 1, bias=False)

        # ── 输出投影 ──
        # SPD 折叠后 4C → C_out，混合 4 个偏移位置的信息
        self.conv_out = nn.Conv2d(in_channels * 4, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, \
            f"SCA-Block requires even H,W, got ({H}, {W})"

        # ═══════════════════════════════════════════════════
        # Step 1: CA 在原始分辨率上计算 1D 方向注意力
        # ═══════════════════════════════════════════════════

        # H 方向全局平均池化: (B, C, H, 1)
        # 保留行结构信息（对水平方向的条状缺陷敏感）
        x_h = F.adaptive_avg_pool2d(x, (H, 1))

        # W 方向全局平均池化: (B, C, 1, W)
        # 保留列结构信息（对垂直方向的条状缺陷敏感）
        x_w = F.adaptive_avg_pool2d(x, (1, W))

        # 共享 1×1 Conv 降维: (B, C, H, 1) → (B, r, H, 1)
        x_h = self.relu(self.bn_reduce(self.conv_ca_reduce(x_h)))
        # (B, C, 1, W) → (B, r, 1, W)
        x_w = self.relu(self.bn_reduce(self.conv_ca_reduce(x_w)))

        # ═══════════════════════════════════════════════════
        # Step 2: 对注意力做 SPD 折叠 —— 融合关键
        # 【融合逻辑】CA 的 1D 池化在原始分辨率生成后，与特征
        #  经历相同的 SPD 空间折叠过程，确保注意力权重与折叠后
        #  的每个子像素位置精确对齐。
        # ═══════════════════════════════════════════════════

        # ── H 方向注意力 SPD 折叠 ──
        # (B, r, H, 1) → 隔行采样 → (B, r*2, H/2, 1)
        # 偶数行对应 2×2 块的上半部分，奇数行对应下半部分
        x_h_even = x_h[:, :, 0::2, :]      # (B, r, H/2, 1)
        x_h_odd  = x_h[:, :, 1::2, :]      # (B, r, H/2, 1)
        x_h_spd  = torch.cat([x_h_even, x_h_odd], dim=1)  # (B, r*2, H/2, 1)

        # ── W 方向注意力 SPD 折叠 ──
        # (B, r, 1, W) → 隔列采样 → (B, r*2, 1, W/2)
        # 偶数列对应 2×2 块的左半部分，奇数列对应右半部分
        x_w_even = x_w[:, :, :, 0::2]      # (B, r, 1, W/2)
        x_w_odd  = x_w[:, :, :, 1::2]      # (B, r, 1, W/2)
        x_w_spd  = torch.cat([x_w_even, x_w_odd], dim=1)  # (B, r*2, 1, W/2)

        # 将折叠后的注意力扩张恢复通道 + Sigmoid 归一化
        att_h = torch.sigmoid(self.conv_expand_h(x_h_spd))  # (B, C, H/2, 1)
        att_w = torch.sigmoid(self.conv_expand_w(x_w_spd))  # (B, C, 1, W/2)

        # ═══════════════════════════════════════════════════
        # Step 3: 对原始特征做 SPD 折叠
        # ═══════════════════════════════════════════════════

        # 将 H×W 空间网格的 2×2 像素块拆分为 4 个子特征图
        # 每个子图对应一个子像素偏移位置
        x_00 = x[:, :, 0::2, 0::2]      # 左上: (B, C, H/2, W/2)
        x_01 = x[:, :, 0::2, 1::2]      # 右上: (B, C, H/2, W/2)
        x_10 = x[:, :, 1::2, 0::2]      # 左下: (B, C, H/2, W/2)
        x_11 = x[:, :, 1::2, 1::2]      # 右下: (B, C, H/2, W/2)

        # 在通道维度拼接: (B, 4C, H/2, W/2)
        # 4C 通道分别对应 4 个空间偏移位置的特征
        x_spd = torch.cat([x_00, x_01, x_10, x_11], dim=1)

        # ═══════════════════════════════════════════════════
        # Step 4: 融合注意力 + 输出投影
        # ═══════════════════════════════════════════════════

        # 注意力重复 4 次以匹配 4C 通道
        # 语义：每个子像素偏移位置共享相同的方向注意力权重
        att_h = att_h.repeat(1, 4, 1, 1)  # (B, 4C, H/2, 1)
        att_w = att_w.repeat(1, 4, 1, 1)  # (B, 4C, 1, W/2)

        # 方向注意力加权：每行/每列的重要性独立调制每个子像素
        x_weighted = x_spd * att_h * att_w  # (B, 4C, H/2, W/2)

        # 输出投影：混合 4 个偏移位置的信息，降维到 C_out
        x_out = self.conv_out(x_weighted)   # (B, C_out, H/2, W/2)
        x_out = self.relu(self.bn_out(x_out))

        return x_out


if __name__ == "__main__":
    # ── 测试 SCA-Block 正向传播维度对齐 ──

    print("=" * 60)
    print("SCA-Block 维度测试")
    print("=" * 60)

    # 测试用例 1: 标准输入 (batch=2, C=32, H=64, W=64)
    B, C, H, W = 2, 32, 64, 64
    x = torch.randn(B, C, H, W)
    model = SCABlock(in_channels=C, out_channels=64)
    out = model(x)

    print(f"\n测试 1:")
    print(f"  输入: {list(x.shape)}  ← (B={B}, C={C}, H={H}, W={W})")
    print(f"  输出: {list(out.shape)} ← (B={B}, C=64, H={H//2}, W={W//2})")

    assert out.shape == (B, 64, H // 2, W // 2), \
        f"Shape mismatch: {out.shape}"
    print(f"  [OK] 维度对齐通过")

    # 测试用例 2: 不同通道数 (C=64, C_out=128)
    B2, C2, H2, W2 = 2, 64, 128, 128
    x2 = torch.randn(B2, C2, H2, W2)
    model2 = SCABlock(in_channels=C2, out_channels=128)
    out2 = model2(x2)

    print(f"\n测试 2 (不同通道数):")
    print(f"  输入: {list(x2.shape)}")
    print(f"  输出: {list(out2.shape)}")
    assert out2.shape == (B2, 128, H2 // 2, W2 // 2), \
        f"Shape mismatch: {out2.shape}"
    print(f"  [OK] 维度对齐通过")

    # 测试用例 3: 动态 batch 支持
    B3 = 4
    x3 = torch.randn(B3, 32, 64, 64)
    out3 = model(x3)  # 复用 model1（C_in=32, C_out=64）
    print(f"\n测试 3 (动态 batch):")
    print(f"  输入: {list(x3.shape)}")
    print(f"  输出: {list(out3.shape)}")
    assert out3.shape == (B3, 64, 32, 32), \
        f"Shape mismatch: {out3.shape}"
    print(f"  [OK] 动态 Batch Size 支持通过")

    # 测试用例 4: 梯度回传验证
    x4 = torch.randn(2, 32, 64, 64, requires_grad=True)
    out4 = model(x4)
    loss = out4.sum()
    loss.backward()
    has_grad = x4.grad is not None and x4.grad.abs().sum().item() > 0
    print(f"\n测试 4 (梯度回传):")
    print(f"  loss = {loss.item():.4f}")
    print(f"  grad norm = {x4.grad.abs().sum().item():.6f}")
    assert has_grad, "No gradient flowing back!"
    print(f"  [OK] 梯度回传通过")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[SCA-Block 参数统计]:")
    print(f"  总参数量:   {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  Flops (估计): ~{4 * C * 64 * 64 * H//2 * W//2 / 1e6:.1f}M")  # 粗略
    print(f"\n{'=' * 60}")
    print(f"[PASS] 所有测试通过！SCA-Block 维度正确，梯度正常回传。")
    print(f"{'=' * 60}")
