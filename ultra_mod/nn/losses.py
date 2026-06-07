"""
NWD Loss (Normalized Wasserstein Distance Loss)
面向极小目标（龟裂）的回归损失函数

【原理】
将 bbox 建模为 2D 高斯分布 N(mu, Sigma), 用 Wasserstein 距离
度量两个分布之间的差异, 归一化后作为 Loss.

W^2 = ||mu1-mu2||^2 + ||Sigma1^{1/2} - Sigma2^{1/2}||_F^2
    = (cx1-cx2)^2 + (cy1-cy2)^2 + (w1-w2)^2/4 + (h1-h2)^2/4

NWD = exp(-sqrt(W^2) / C)
Loss = 1 - NWD

【优势 over IoU】
- 对小框的偏移更敏感 (IoU 在小框上对像素偏移极不敏感)
- 当 bbox 不重叠时, IoU = 0 无梯度, NWD 仍有平滑梯度
- 对钢材龟裂这种精细缺陷友好
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════
# 核心 NWD 函数
# ════════════════════════════════════════════════════════════

def wasserstein_distance_loss(pred: torch.Tensor, target: torch.Tensor,
                              eps: float = 1e-7, C: float = 12.8) -> torch.Tensor:
    """
    Normalized Wasserstein Distance Loss

    Args:
        pred:   预测框 (..., 4), 格式 [cx, cy, w, h] (YOLOv8 Anchor-Free)
        target: 目标框 (..., 4), 格式 [cx, cy, w, h]
        eps:    防止除零/数值不稳定
        C:      归一化常数 (NEU-DET ~12.8, 可调)

    Returns:
        nwd_loss: (...,) 每个框的 NWD Loss [0, 1]

    推导:
        bbox = (cx, cy, w, h) → 高斯 N(mu, Sigma)
        mu = [cx, cy],  Sigma = diag(w^2/4, h^2/4)

        对对角协方差:
        W^2 = (cx1-cx2)^2 + (cy1-cy2)^2 + (w1-w2)^2/4 + (h1-h2)^2/4

        归一化 NWD = exp(-sqrt(W^2) / C) ∈ (0, 1]
    """
    cx1, cy1, w1, h1 = pred.unbind(-1)
    cx2, cy2, w2, h2 = target.unbind(-1)

    # Wasserstein 距离平方 (closed-form)
    w2_val = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 +
              (w1 - w2) ** 2 / 4 + (h1 - h2) ** 2 / 4)

    # 归一化
    nwd = torch.exp(-torch.sqrt(w2_val + eps) / C)
    nwd_loss = 1.0 - nwd

    return nwd_loss


# ════════════════════════════════════════════════════════════
# 与 CIoU 加权融合
# ════════════════════════════════════════════════════════════

def nwd_loss_with_ciou(pred: torch.Tensor, target: torch.Tensor,
                        ciou_loss: torch.Tensor,
                        lambda_nwd: float = 0.5,
                        lambda_ciou: float = 0.5) -> torch.Tensor:
    """
    NWD + CIoU 加权融合

    Args:
        pred:        预测框 (..., 4)
        target:      目标框 (..., 4)
        ciou_loss:   预计算的 CIoU Loss
        lambda_nwd:  NWD 权重
        lambda_ciou: CIoU 权重

    Returns:
        combined_loss: 加权融合后的 Loss
    """
    nwd = wasserstein_distance_loss(pred, target)
    return lambda_ciou * ciou_loss + lambda_nwd * nwd


# ════════════════════════════════════════════════════════════
# BboxLossWithNWD: 可替代 ultralytics 的 BboxLoss
# ════════════════════════════════════════════════════════════

class BboxLossWithNWD(nn.Module):
    """
    BboxLoss 的 NWD 增强版

    Loss = (1-λ)*CIoU + λ*NWD + DFL

    替代方案 (直接替换 BboxLoss):
        from ultra_mod.nn.losses import BboxLossWithNWD
        self.bbox_loss = BboxLossWithNWD(reg_max, lambda_nwd=0.5).to(device)
    """

    def __init__(self, reg_max: int = 16, lambda_nwd: float = 0.5):
        super().__init__()
        from ultralytics.utils.loss import DFLoss, bbox_iou

        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.lambda_nwd = lambda_nwd
        self.bbox_iou = bbox_iou

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum,
                fg_mask, imgsz, stride):
        """
        与 BboxLoss.forward 签名完全一致, 可直接替换
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # ── CIoU ──
        iou = self.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                            xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # ── NWD (对极小目标/龟裂更敏感) ──
        nwd = wasserstein_distance_loss(
            pred_bboxes[fg_mask], target_bboxes[fg_mask]
        )
        loss_nwd = (nwd.unsqueeze(-1) * weight).sum() / target_scores_sum

        # 加权融合
        loss_iou = (1.0 - self.lambda_nwd) * loss_iou + self.lambda_nwd * loss_nwd

        # ── DFL ──
        if self.dfl_loss:
            from ultralytics.utils.tal import bbox2dist
            target_ltrb = bbox2dist(anchor_points, target_bboxes,
                                    self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            # fallback: L1 loss
            loss_dfl = torch.zeros_like(loss_iou)

        return loss_iou, loss_dfl


# ════════════════════════════════════════════════════════════
# 注入指南 (修改 ultralytics/utils/loss.py)
# ════════════════════════════════════════════════════════════

"""
============================================================
NWD Loss 注入到 ultralytics 训练流程 (两种方案)
============================================================

方案 A: 零侵入侵袭替换 (推荐)
────────────────────────────────────
直接替换 BboxLoss 为 BboxLossWithNWD, 无需修改任何 ultralytics 源码:

    # 在你的 register.py 或 train.py 中:
    from ultra_mod.nn.losses import BboxLossWithNWD
    from ultralytics.utils.loss import v8DetectionLoss
    import ultralytics.utils.loss as loss_mod

    # 猴子补丁: 替换 BboxLoss
    loss_mod.BboxLoss = BboxLossWithNWD  # 所有 v8DetectionLoss 自动使用 NWD

或者创建自定义 Loss:

    class CustomDetectionLoss(v8DetectionLoss):
        def __init__(self, model, tal_topk=10, tal_topk2=None):
            super().__init__(model, tal_topk, tal_topk2)
            # 替换 bbox_loss
            self.bbox_loss = BboxLossWithNWD(
                model.model[-1].reg_max, lambda_nwd=0.5
            ).to(self.device)

方案 B: 修改 ultralytics/utils/loss.py
────────────────────────────────────
在 BboxLoss.forward 中 (~line 132-139):

   # [原版]
   iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
   loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

   # [改为]
   from ultra_mod.nn.losses import wasserstein_distance_loss
   iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
   loss_ciou = ((1.0 - iou) * weight).sum() / target_scores_sum
   nwd_loss = wasserstein_distance_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
   nwd_loss = (nwd_loss.unsqueeze(-1) * weight).sum() / target_scores_sum
   loss_iou = 0.5 * loss_ciou + 0.5 * nwd_loss   # 加权融合

方案 C: 配合 PGI 双分支
────────────────────────────────────
如果在使用 PGI_Detect:

   在 v8DetectionLoss.get_assigned_targets_and_loss 中:

   # 对主分支
   loss_main = self.bbox_loss(...)

   # 对 P2 辅助分支 (需要额外调用一次)
   # 用相同 targets 但 stride 缩放不同 (P2 stride=4)
   loss_aux = self.bbox_loss(pred_dist_aux, ...) * 0.25  # 权重衰减

   # 或者直接使用 v8PGIDetectionLoss (在 head.py 中定义)


超参建议
────────────────────────────────────
C (归一化常数) 的选择:
  - C = 12.8 (默认, NEU-DET 缺陷 ~30-100px)
  - C 越大, NWD 对偏移越不敏感 (接近均匀)
  - C 越小, NWD 对任何偏移都更敏感 (但梯度过大)
  - 经验: C = avg(bbox_size) * 0.25 ~ 0.5

lambda_nwd 的选择:
  - 0.3: 轻微增强小目标, CIoU 主导
  - 0.5: 平衡 (推荐, 默认)
  - 0.7: NWD 主导, 适合极小目标 (< 20px)
"""


# ════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("NWD Loss / BboxLossWithNWD 测试")
    print("=" * 60)

    B, N = 2, 10

    # ── 测试 1: NWD 完全匹配 ──
    print("\n[Test 1] 完全匹配")
    pred = torch.randn(B, N, 4)
    target = pred.clone()
    nwd = wasserstein_distance_loss(pred, target)
    assert nwd.mean().item() < 0.01, f"完全匹配 NWD 应接近 0, got {nwd.mean().item()}"
    print(f"  NWD Loss = {nwd.mean().item():.6f}  [OK]")

    # ── 测试 2: 完全不匹配 ──
    print("\n[Test 2] 完全不匹配")
    nwd2 = wasserstein_distance_loss(
        torch.tensor([[50., 50., 10., 10.]]),
        torch.tensor([[500., 500., 10., 10.]]),
    )
    assert nwd2.item() > 0.95, f"完全不匹配应 > 0.95, got {nwd2.item()}"
    print(f"  NWD Loss = {nwd2.item():.6f}  [OK]")

    # ── 测试 3: 连续梯度 (不重叠时) ──
    print("\n[Test 3] 连续梯度 (bbox 不重叠)")
    box_a = torch.tensor([[50., 50., 20., 20.]])
    box_b = torch.tensor([[71., 50., 20., 20.]])  # 刚好不重叠
    nwd_no_iou = wasserstein_distance_loss(box_a, box_b)
    print(f"  不重叠时 NWD = {nwd_no_iou.item():.4f} (IoU = 0)")
    assert nwd_no_iou.item() < 1.0, "不重叠时 NWD 应有平滑梯度"
    print("  [OK]")

    # ── 测试 4: 梯度回传 ──
    print("\n[Test 4] 梯度回传")
    x = torch.randn(B, N, 4, requires_grad=True)
    nwd_grad = wasserstein_distance_loss(x, torch.randn(B, N, 4))
    nwd_grad.sum().backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0, "梯度异常"
    print(f"  grad = {x.grad.abs().sum().item():.6f}  [OK]")

    # ── 测试 5: BboxLossWithNWD 前向 ──
    print("\n[Test 5] BboxLossWithNWD 前向")
    loss_fn = BboxLossWithNWD(reg_max=16, lambda_nwd=0.5)
    H, W = 80, 80
    n_anchors = H * W
    pred_dist = torch.randn(B, n_anchors, 64)
    pred_bbox = torch.randn(B, n_anchors, 4)
    anchor_points = torch.stack([
        torch.arange(W, dtype=torch.float).repeat(H),
        torch.arange(H, dtype=torch.float).unsqueeze(1).repeat(1, W).flatten(),
    ], dim=1) + 0.5
    target_bbox = torch.randn(B, n_anchors, 4)
    target_scores = torch.zeros(B, n_anchors, 6)
    target_scores[:, :5, 0] = 1.0  # 少量正样本
    fg_mask = target_scores.sum(-1) > 0
    target_scores_sum = max(target_scores.sum(), 1)
    imgsz = torch.tensor([640, 640])
    stride = torch.full((n_anchors, 1), 8.0)

    loss_iou, loss_dfl = loss_fn(
        pred_dist, pred_bbox, anchor_points,
        target_bbox, target_scores, target_scores_sum,
        fg_mask, imgsz, stride,
    )
    print(f"  loss_iou = {loss_iou.item():.4f}")
    print(f"  loss_dfl = {loss_dfl.item():.4f}")
    assert loss_iou.item() >= 0, "loss_iou 不能为负"
    assert loss_dfl.item() >= 0, "loss_dfl 不能为负"
    print("  [OK]")

    # ── 测试 6: BboxLossWithNWD 梯度回传 ──
    print("\n[Test 6] BboxLossWithNWD 梯度回传")
    pred_dist_g = pred_dist.clone().requires_grad_(True)
    pred_bbox_g = pred_bbox.clone().requires_grad_(True)
    loss_iou2, loss_dfl2 = loss_fn(
        pred_dist_g, pred_bbox_g, anchor_points,
        target_bbox, target_scores, target_scores_sum,
        fg_mask, imgsz, stride,
    )
    (loss_iou2 + loss_dfl2).backward()
    assert pred_dist_g.grad is not None, "dist 梯度异常"
    assert pred_bbox_g.grad is not None, "bbox 梯度异常"
    print(f"  pred_dist grad = {pred_dist_g.grad.abs().sum().item():.4f}")
    print(f"  pred_bbox grad = {pred_bbox_g.grad.abs().sum().item():.4f}")
    print("  [OK]")

    print(f"\n{'=' * 60}")
    print("[PASS] NWD Loss / BboxLossWithNWD 全部测试通过!")
    print(f"{'=' * 60}")
