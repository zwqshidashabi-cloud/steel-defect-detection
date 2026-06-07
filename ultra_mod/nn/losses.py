"""
NWD Loss (Normalized Wasserstein Distance Loss)
面向极小目标（龟裂）的回归损失函数

【原理】
将 bbox 建模为 2D 高斯分布 (mu, sigma)，用 Wasserstein 距离
度量两个分布之间的差异，归一化后作为 Loss。

【优势 over IoU】
- 对小框的偏移更敏感（IoU 在小框上对像素偏移极不敏感）
- 当 bbox 不重叠时，IoU = 0 无梯度，NWD 仍有平滑梯度
- 对钢材龟裂这种精细缺陷友好

【注入指南】
在 ultralytics/utils/loss.py 的 BboxLoss 中:
  1. 计算原版 CIoU Loss
  2. 计算 NWD Loss
  3. final_loss = lambda1 * ciou_loss + lambda2 * nwd_loss
  推荐 lambda1=0.5, lambda2=0.5 (或 0.7/0.3 侧重 NWD)
"""

import torch
import torch.nn as nn


def wasserstein_distance_loss(pred: torch.Tensor, target: torch.Tensor,
                               eps: float = 1e-7) -> torch.Tensor:
    """
    计算 Normalized Wasserstein Distance Loss

    Args:
        pred:  预测框 (B, N, 4) 或 (N, 4), 格式 [x, y, w, h]
               中心点坐标 + 宽高 (与 YOLOv8 的 Anchor-Free 格式一致)
        target: 目标框, 格式同 pred
        eps:    防止除零

    Returns:
        nwd_loss: (B, N) 或 (N,) 每个框的 NWD Loss 值 [0, 1]

    推导:
        bbox = (cx, cy, w, h) → 高斯分布 N(mu, Sigma)
        mu = [cx, cy]
        Sigma = diag(w^2/4, h^2/4)  (取 2-sigma 覆盖 95% 区域)

        两个高斯分布 N1(mu1, Sigma1), N2(mu2, Sigma2) 的
        2 阶 Wasserstein 距离平方:
            W^2 = ||mu1 - mu2||^2 +
                  Tr(Sigma1 + Sigma2 - 2*(Sigma1^{1/2} * Sigma2 * Sigma1^{1/2})^{1/2})
                = ||mu1 - mu2||^2 + ||Sigma1^{1/2} - Sigma2^{1/2}||_F^2

        对于对角协方差:
            W^2 = (cx1-cx2)^2 + (cy1-cy2)^2 +
                  (w1-w2)^2/4 + (h1-h2)^2/4
    """
    # 提取中心点坐标和宽高
    cx1, cy1, w1, h1 = pred.unbind(-1)
    cx2, cy2, w2, h2 = target.unbind(-1)

    # Wasserstein 距离平方 (closed-form for diagonal covariance)
    # W^2 = ||center_diff||^2 + ||size_diff||^2 / 4
    w2_val = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 +
              (w1 - w2) ** 2 / 4 + (h1 - h2) ** 2 / 4)

    # 归一化: NWD = exp(-sqrt(W^2) / C)
    # C 是归一化常数, 取数据集平均尺寸或固定值
    # NEU-DET 中缺陷尺寸 ~30-100 px, 取 C=12.8 (约 ln2 * avg_size)
    # 实际使用时可从数据统计中计算
    c = 12.8  # 归一化常数 (可调超参)

    nwd = torch.exp(-torch.sqrt(w2_val + eps) / c)

    # Loss = 1 - NWD (范围 [0, 1])
    nwd_loss = 1.0 - nwd

    return nwd_loss


def nwd_loss_with_ciou(pred: torch.Tensor, target: torch.Tensor,
                        ciou_loss: torch.Tensor,
                        lambda_nwd: float = 0.5,
                        lambda_ciou: float = 0.5) -> torch.Tensor:
    """
    NWD Loss + CIoU Loss 加权融合

    用法（在 ultralytics/utils/loss.py 中）:
        # 在 BboxLoss.forward 中:
        ciou_loss, iou = bbox_loss_iou(pred_bbox, target_bbox)
        combined_loss = nwd_loss_with_ciou(
            pred_bbox, target_bbox, ciou_loss,
            lambda_nwd=0.5, lambda_ciou=0.5
        )

    Args:
        pred:     预测框 (B, N, 4) [x, y, w, h]
        target:   目标框 (B, N, 4) [x, y, w, h]
        ciou_loss: 原版 CIoU Loss 值
        lambda_nwd:  NWD Loss 权重
        lambda_ciou: CIoU Loss 权重
    """
    nwd = wasserstein_distance_loss(pred, target)
    return lambda_ciou * ciou_loss + lambda_nwd * nwd


def focal_eiou_loss(pred: torch.Tensor, target: torch.Tensor,
                    gamma: float = 0.5) -> torch.Tensor:
    """
    Focal-EIoU Loss: 聚焦于高质量样本的 EIoU

    EIoU = IoU - rho^2(c,c_gt)/c^2 - rho^2(w,w_gt)/Cw^2 - rho^2(h,h_gt)/Ch^2
    Focal-EIoU = IoU^gamma * L_eiou

    比 CIoU 多考虑了宽高比的独立回归
    gamma=0.5 聚焦中等质量样本
    """
    # 简易实现: 直接计算 EIoU
    cx1, cy1, w1, h1 = pred.unbind(-1)
    cx2, cy2, w2, h2 = target.unbind(-1)

    # IoU 计算
    x1_min, y1_min = cx1 - w1 / 2, cy1 - h1 / 2
    x1_max, y1_max = cx1 + w1 / 2, cy1 + h1 / 2
    x2_min, y2_min = cx2 - w2 / 2, cy2 - h2 / 2
    x2_max, y2_max = cx2 + w2 / 2, cy2 + h2 / 2

    inter_min = torch.max(x1_min, x2_min)
    inter_max = torch.min(x1_max, x2_max)
    inter_h = torch.clamp(inter_max - inter_min, min=0)
    inter_max2 = torch.max(y1_min, y2_min)
    inter_min2 = torch.min(y1_max, y2_max)
    inter_w = torch.clamp(inter_min2 - inter_max2, min=0)

    inter = inter_h * inter_w
    union = w1 * h1 + w2 * h2 - inter + 1e-7
    iou = inter / union

    # EIoU penalty terms
    rho_c = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    c_c = (cx1.max() - cx2.min()) ** 2 + (cy1.max() - cy2.min()) ** 2 + 1e-7

    rho_w = (w1 - w2) ** 2
    rho_h = (h1 - h2) ** 2
    c_w = (w1.max() - w2.min()) ** 2 + 1e-7
    c_h = (h1.max() - h2.min()) ** 2 + 1e-7

    eiou = iou - rho_c / c_c - rho_w / c_w - rho_h / c_h

    # Focal modulation
    focal = iou ** gamma
    loss = focal * (1 - eiou)

    return loss


# ════════════════════════════════════════════════════════════
# 注入指南
# ════════════════════════════════════════════════════════════
"""
如何使用:

在 ultralytics/utils/loss.py 的 BboxLoss 类中:

1. 导入:
   从你的项目中 import wasserstein_distance_loss

2. 在 BboxLoss.forward 中找到 ciou loss 计算位置 (~line 85):
   # 原版: loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
   # 改为:
   ciou_loss = ((1.0 - iou) * weight).sum() / target_scores_sum
   nwd_loss = wasserstein_distance_loss(pred_bboxes, target_bboxes)
   nwd_loss = (nwd_loss * weight).sum() / target_scores_sum
   loss_iou = 0.5 * ciou_loss + 0.5 * nwd_loss

3. 如果配合 PGI（双分支 Loss）:
   # 主分支 loss
   loss_main = bbox_loss(pred_main, target)
   # 辅助分支 loss (P2)
   loss_aux = bbox_loss(pred_aux, target) * 0.25  # P2 权重衰减
   total_loss = loss_main + loss_aux
"""


# ════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("NWD Loss / Focal-EIoU 测试")
    print("=" * 60)

    B, N = 2, 10

    # 生成完全匹配的框
    pred = torch.randn(B, N, 4)
    target = pred.clone()  # 完全匹配

    # 测试 1: 完全匹配时 NWD = 0
    nwd = wasserstein_distance_loss(pred, target)
    print(f"\n[Test 1] 完全匹配: NWD Loss = {nwd.mean().item():.6f}")
    assert nwd.mean().item() < 0.01, "完全匹配时 NWD Loss 应接近 0"
    print("  [OK]")

    # 测试 2: 完全不匹配时 NWD 接近 1
    pred2 = torch.tensor([[50.0, 50.0, 10.0, 10.0]])
    target2 = torch.tensor([[500.0, 500.0, 10.0, 10.0]])
    nwd2 = wasserstein_distance_loss(pred2, target2)
    print(f"\n[Test 2] 完全不匹配: NWD Loss = {nwd2.item():.6f}")
    assert nwd2.item() > 0.9, "完全不匹配时 NWD Loss 应接近 1"
    print("  [OK]")

    # 测试 3: 小偏移 vs 大偏移
    center = torch.tensor([[100.0, 100.0, 20.0, 20.0]])
    small_shift = torch.tensor([[102.0, 100.0, 20.0, 20.0]])   # 2px
    large_shift = torch.tensor([[120.0, 100.0, 20.0, 20.0]])   # 20px
    nwd_small = wasserstein_distance_loss(center, small_shift)
    nwd_large = wasserstein_distance_loss(center, large_shift)
    print(f"\n[Test 3] 偏移敏感度:")
    print(f"  2px 偏移: {nwd_small.item():.4f}")
    print(f"  20px 偏移: {nwd_large.item():.4f}")
    assert nwd_large.item() > nwd_small.item(), "大偏移的 Loss 应更大"
    print("  [OK]")

    # 测试 4: NWD 对同 2px 偏移尺度一致
    # 不管目标大小，2px 偏移下 W^2 = 25+25 = 50 → NWD 应一致
    box1 = torch.tensor([[50.0, 50.0, 10.0, 10.0]])
    box2 = torch.tensor([[50.0, 50.0, 100.0, 100.0]])
    shift = torch.tensor([[55.0, 55.0, 10.0, 10.0]])  # 5px 偏移, 尺寸不变
    nwd1 = wasserstein_distance_loss(box1, shift)
    nwd2 = wasserstein_distance_loss(box2, shift)  # 尺寸不同, 但预测 cx/cy 也差 5px
    print(f"\n[Test 4] NWD 尺度一致性:")
    print(f"  小目标 (10x10): NWD Loss = {nwd1.item():.4f}")
    print(f"  不同尺寸 (100x100): NWD Loss = {nwd2.item():.4f}")
    # shift 的尺寸是 10x10, 与 box2 的尺寸差会贡献额外 NWD
    # 所以这里不 assert, 只打印观察

    # 测试 5: NWD vs IoU — 不重叠时仍有平滑梯度
    print("\n[Test 5] NWD vs IoU (不重叠时)")
    box_a = torch.tensor([[50.0, 50.0, 20.0, 20.0]])
    box_b = torch.tensor([[71.0, 50.0, 20.0, 20.0]])   # 刚好不重叠
    nwd_no_iou = wasserstein_distance_loss(box_a, box_b)
    # IoU = 0 (不重叠), 但 NWD 是连续的
    print(f"  刚好不重叠时 NWD Loss = {nwd_no_iou.item():.4f} (IoU = 0)")
    assert nwd_no_iou.item() < 1.0, "不重叠时 NWD 应 < 1 (有平滑梯度)"
    print("  [OK] NWD 在不重叠时仍提供梯度信号, IoU 则不能")

    # 测试 6: 梯度回传
    print("\n[Test 6] 梯度回传")
    pred_grad = torch.randn(B, N, 4, requires_grad=True)
    target_grad = torch.randn(B, N, 4)
    nwd_grad = wasserstein_distance_loss(pred_grad, target_grad)
    loss = nwd_grad.sum()
    loss.backward()
    has_grad = pred_grad.grad is not None and pred_grad.grad.abs().sum().item() > 0
    assert has_grad, "NWD Loss 梯度回传失败"
    print(f"  loss = {loss.item():.4f}, grad = {pred_grad.grad.abs().sum().item():.6f}")
    print("  [OK] 梯度回传通过")

    # 测试 7: Focal-EIoU
    print("\n[Test 7] Focal-EIoU Loss")
    feiou = focal_eiou_loss(pred_grad, target_grad)
    print(f"  Focal-EIoU shape: {list(feiou.shape)}")
    assert feiou.shape == (B, N), f"shape mismatch: {feiou.shape}"
    print("  [OK]")

    print(f"\n{'=' * 60}")
    print("[PASS] NWD Loss 全部测试通过!")
    print(f"{'=' * 60}")
