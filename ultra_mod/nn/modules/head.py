"""
PGI_Detect: Programmable Gradient Information 检测头
继承 ultralytics Detect 以实现完整的训练/推理/导出兼容

【设计要点】
1. 继承 Detect → isinstance(m, Detect) 通过 → stride 自动计算、bias_init、device 转换
2. 训练: P2 辅助分支 + 主分支联合计算, 返回 dict 含 "aux" 字段
3. 推理: 完全等效原版 Detect, 零 P2 开销
4. 导出: 通过 export=True 标记剥离 P2

【使用方式】
head:
  - [[15, 12, 9, 18], 1, PGI_Detect, [nc]]

ch 传入顺序: [P3_ch, P4_ch, P5_ch, P2_ch]
"""

import copy
import math

import torch
import torch.nn as nn

# ── 使用 Ultralytics 官方组件以保证完全兼容 ──
# 注意: 实际运行时通过 register.py 注入到 ultralytics 命名空间,
#       这些 import 会在 ultralytics 上下文中解析
from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors


class PGI_Detect(Detect):
    """
    PGI 检测头 — 训练时 P2 辅助监督, 推理零额外算力

    Args:
        nc: 类别数 (NEU-DET = 6)
        reg_max: DFL 通道数 (default=16)
        end2end: 端到端模式 (不启用)
        ch: 各检测层通道列表 [P3, P4, P5, P2]
            前 3 个为主检测头, 最后 1 个为 P2 辅助

    Attributes:
        nl: 主分支检测层数 (= len(ch)-1)
        aux_cv2: P2 辅助分支 box 回归
        aux_cv3: P2 辅助分支 cls 分类
    """

    export = False          # 导出模式标记 (enabled by switch_to_export)
    aux_weight = 0.25       # P2 辅助 Loss 权重衰减

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        # ch = [c3, c4, c5, c2] → 前 3 主分支, 最后 1 P2
        assert len(ch) >= 4, (
            f"PGI_Detect requires at least 4 channel values [P3,P4,P5,P2], got {len(ch)}"
        )
        self._main_nl = len(ch) - 1  # 主分支层数 = 3
        main_ch = ch[:self._main_nl]   # [c3, c4, c5]
        aux_ch = ch[self._main_nl]     # c2 (P2)

        # ── 初始化父类 Detect (仅主分支) ──
        # Detect.__init__ 设置:
        #   self.nl = len(ch) = self._main_nl = 3
        #   self.cv2, self.cv3 = 主分支的 box/cls 卷积
        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=main_ch)
        self.nl = self._main_nl  # 明确: nl 仅计主分支

        # ── 从父类 cv2/cv3 导出 c2/c3 通道数 ──
        # cv2[i] 的最后 conv: in_channels=c2, out_channels=4*reg_max
        # cv3[i] 的最后 conv: in_channels=c3, out_channels=nc
        c2 = self.cv2[0][-1].in_channels
        c3 = self.cv3[0][-1].in_channels

        # ── P2 辅助分支 (单层) ──
        # 结构同 Detect 的单层: DWConv → Conv × 2, 最后 1×1 conv
        self.aux_cv2 = nn.Sequential(
            Conv(aux_ch, c2, 3),
            Conv(c2, c2, 3),
            nn.Conv2d(c2, 4 * self.reg_max, 1),
        )
        self.aux_cv3 = nn.Sequential(
            nn.Sequential(DWConv(aux_ch, aux_ch, 3), Conv(aux_ch, c3, 1)),
            nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
            nn.Conv2d(c3, self.nc, 1),
        )

        # ── 重置 stride: 父类 Detect 为 3 层计算 stride ──
        # 实际 stride 在 DetectionModel.build 中通过前向计算覆盖
        self.stride = torch.zeros(self.nl)

    # ──────────────── Forward ────────────────

    def forward(self, x):
        """
        Args:
            x: list of 4 feature maps [P3, P4, P5, P2]

        Returns:
            训练: dict {
                "boxes":   (B, 4*reg_max, total_anchors),
                "scores":  (B, nc, total_anchors),
                "feats":   [P3_feat, P4_feat, P5_feat],
                "aux":     {"boxes": ..., "scores": ..., "feats": [P2_feat]}
            }
            推理: tuple (y, preds)   (与原生 Detect 一致)
            导出: y (仅解码框, 与原生 Detect 一致)
        """
        # 分离主分支和辅助分支输入
        main_feats = x[:self._main_nl]      # [P3, P4, P5]
        aux_feats = x[self._main_nl:]       # [P2]

        # ── 主分支预测 ──
        main_preds = self.forward_head(main_feats, self.cv2, self.cv3)

        # ── 训练模式: 额外计算 P2 辅助分支 ──
        if self.training and not self.export:
            aux_preds = self._forward_aux(aux_feats)
            main_preds["aux"] = aux_preds
            return main_preds

        # ── 推理/导出: 纯 Detect 行为, P2 零开销 ──
        y = self._inference(main_preds)
        return y if self.export else (y, main_preds)

    def _forward_aux(self, x):
        """P2 辅助分支前向"""
        assert len(x) == 1, f"aux expects 1 feature map, got {len(x)}"
        bs = x[0].shape[0]
        boxes = self.aux_cv2(x[0]).view(bs, 4 * self.reg_max, -1)
        scores = self.aux_cv3(x[0]).view(bs, self.nc, -1)
        return {"boxes": boxes, "scores": scores, "feats": list(x)}

    # ──────────────── Bias Init ────────────────

    def bias_init(self):
        """
        初始化检测头偏置 (扩展父类以包含 aux 分支)

        父类 Detect.bias_init 初始化 self.cv2 和 self.cv3,
        我们额外初始化 self.aux_cv2 和 self.aux_cv3.
        """
        super().bias_init()

        # 初始化 aux box 分支
        self.aux_cv2[-1].bias.data[:] = 2.0  # box

        # 初始化 aux cls 分支
        cls_last = self.aux_cv3[-1]
        cls_last.bias.data[: self.nc] = math.log(5 / self.nc / (640 / 4) ** 2)  # cls

    # ──────────────── 导出模式 ────────────────

    def switch_to_export(self):
        """
        切换到导出模式:
        训练完成后调用 model.export(...) 时自动触发 export=True,
        此后 P2 辅助分支在前向中被彻底阻断.
        """
        self.export = True

    def switch_to_train(self):
        """恢复到训练模式"""
        self.export = False

    # ──────────────── 序列化兼容 ────────────────

    def _apply(self, fn):
        """Override _apply 确保 aux 相关 buffer 也正确转换设备"""
        self = super()._apply(fn)
        # aux_cv2/cv3 是 nn.Sequential, 其子模块会被父类 _apply 自动处理
        return self


# ════════════════════════════════════════════════════════════
# PGI-Aware Loss: 主分支 + P2 辅助分支 + NWD Loss 联合
# ════════════════════════════════════════════════════════════

class v8PGIDetectionLoss:
    """
    PGI + NWD 增强版检测 Loss

    相比 v8DetectionLoss:
    1. P2 辅助分支 Loss (0.25 倍权重衰减)
    2. BboxLoss 中注入 NWD
    3. 兼容原版 v8DetectionLoss 的 TaskAlignedAssigner

    用法 (替换 DetectionModel.init_criterion):
        def init_criterion(self):
            return v8PGIDetectionLoss(self)
    """

    def __init__(self, model, tal_topk=10, tal_topk2=None, lambda_nwd=0.5):
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.lambda_nwd = lambda_nwd

        from ultralytics.utils.tal import TaskAlignedAssigner
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, num_classes=self.nc,
            alpha=0.5, beta=6.0, stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        from ultra_mod.nn.losses import BboxLossWithNWD as _BboxLossWithNWD
        self.bbox_loss = _BboxLossWithNWD(m.reg_max, lambda_nwd=lambda_nwd).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        from ultralytics.utils.loss import v8DetectionLoss
        # 复用父类方法
        return v8DetectionLoss.preprocess(self, targets, batch_size, scale_tensor)

    def bbox_decode(self, anchor_points, pred_dist):
        from ultralytics.utils.tal import dist2bbox
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype)
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _compute_branch_loss(self, branch_preds, batch):
        """
        对单分支 (main 或 aux) 计算 loss

        Returns:
            loss: [box_loss, cls_loss, dfl_loss]
        """
        loss = torch.zeros(3, device=self.device)
        pred_distri = branch_preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = branch_preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(
            branch_preds["feats"], self.stride, 0.5
        )

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(
                branch_preds["feats"][0].shape[2:], device=self.device, dtype=dtype
            )
            * self.stride[0]
        )

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1),
             batch["cls"].view(-1, 1),
             batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Decode boxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # TAL assigner
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        loss[1] = bce_loss.sum() / target_scores_sum

        # Bbox loss (含 NWD)
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss

    def __call__(self, preds, batch):
        """
        preds: PGI_Detect.forward 输出
               {"boxes":..., "scores":..., "feats":..., "aux": {...}}
               or (y, preds) tuple during eval
        """
        from ultralytics.utils.loss import v8DetectionLoss

        # 推理模式: 走标准 loss 计算 (仅主分支)
        if isinstance(preds, (list, tuple)):
            # 此时 preds = (y, preds_dict) 来自推理模式
            # 标准 v8DetectionLoss 处理
            std_loss = v8DetectionLoss.__new__(v8DetectionLoss)
            # ... 简化: 走标准流程 ...
            # 对于验证集, 使用标准 v8DetectionLoss
            return self._eval_loss(preds, batch)

        # 训练模式: 主分支 + P2 辅助分支
        main_preds = {k: v for k, v in preds.items() if k != "aux"}
        aux_preds = preds.get("aux")

        # 主分支 loss
        main_loss = self._compute_branch_loss(main_preds, batch)

        # P2 辅助分支 loss (带权重衰减)
        total_loss = main_loss.clone()
        if aux_preds is not None:
            aux_loss = self._compute_branch_loss(aux_preds, batch)
            total_loss += aux_loss * getattr(
                self, "aux_weight", 0.25
            )

        loss_sum = total_loss.sum() * batch["img"].shape[0]
        return loss_sum, total_loss.detach()

    def _eval_loss(self, preds, batch):
        """验证时: 取 tuple 中的 preds dict, 只计算主分支"""
        # preds = (y, preds_dict)
        main_preds = preds[1] if isinstance(preds, tuple) else preds
        # 移除 aux (如果在 tuple 中存在)
        if isinstance(main_preds, dict) and "aux" in main_preds:
            main_preds = {k: v for k, v in main_preds.items() if k != "aux"}
        loss = self._compute_branch_loss(main_preds, batch)
        loss_sum = loss.sum() * batch["img"].shape[0]
        return loss_sum, loss.detach()


# ════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PGI_Detect 兼容性测试 (继承 Detect)")
    print("=" * 60)

    B, nc = 2, 6

    # 模拟特征输入: P3(80x80), P4(40x40), P5(20x20), P2(160x160)
    feats = [
        torch.randn(B, 64, 80, 80),
        torch.randn(B, 128, 40, 40),
        torch.randn(B, 256, 20, 20),
        torch.randn(B, 32, 160, 160),
    ]
    ch = (64, 128, 256, 32)

    # ── 测试 1: 构造 ──
    print("\n[Test 1] 构造 PGI_Detect (nc=6, ch=[64,128,256,32])")
    model = PGI_Detect(nc=nc, reg_max=16, ch=ch)
    print(f"  nl (主分支层数)     = {model.nl}")
    print(f"  cv2 长度 (box)      = {len(model.cv2)}")
    print(f"  cv3 长度 (cls)      = {len(model.cv3)}")
    print(f"  has aux_cv2         = {hasattr(model, 'aux_cv2')}")
    print(f"  has aux_cv3         = {hasattr(model, 'aux_cv3')}")
    assert model.nl == 3
    assert len(model.cv2) == 3
    assert len(model.cv3) == 3
    assert isinstance(model, Detect), "PGI_Detect 须继承 Detect"
    print("  [OK]")

    # ── 测试 2: 训练模式输出 ──
    print("\n[Test 2] 训练模式 (model.train())")
    model.train()
    out = model(feats)
    assert isinstance(out, dict), f"训练模式应输出 dict, got {type(out)}"
    assert "boxes" in out, "须含 boxes"
    assert "scores" in out, "须含 scores"
    assert "feats" in out, "须含 feats"
    assert "aux" in out, "须含 aux"
    assert "boxes" in out["aux"], "aux 须含 boxes"
    print(f"  main['boxes'].shape  = {out['boxes'].shape}")
    print(f"  main['scores'].shape = {out['scores'].shape}")
    print(f"  main['feats'] len    = {len(out['feats'])}")
    print(f"  aux['boxes'].shape   = {out['aux']['boxes'].shape}")
    print(f"  aux['feats'] len     = {len(out['aux']['feats'])}")
    assert len(out["feats"]) == 3, "主分支应有 3 层特征"
    assert len(out["aux"]["feats"]) == 1, "aux 应有 1 层特征 (P2)"
    print("  [OK]")

    # ── 测试 3: 推理模式输出 ──
    print("\n[Test 3] 推理模式 (model.eval())")
    model.eval()
    out_eval = model(feats)
    assert isinstance(out_eval, tuple) and len(out_eval) == 2, \
        f"推理模式应输出 tuple(y, preds), got {type(out_eval)}"
    y, preds = out_eval
    print(f"  y.shape        = {y.shape}  ← 解码框 (B, 4+nc, N)")
    print(f"  preds['boxes'] = {preds['boxes'].shape}")
    assert "aux" not in preds, "推理时不应含 aux"
    print("  [OK] 推理模式 P2 已剥离, 零额外算力")

    # ── 测试 4: 导出模式 ──
    print("\n[Test 4] 导出模式 (export=True)")
    model.eval()
    model.switch_to_export()
    out_export = model(feats)
    # export=True: 返回 y (仅解码框), 不是 tuple
    assert isinstance(out_export, torch.Tensor), \
        f"导出模式应输出 tensor, got {type(out_export)}"
    print(f"  export output shape = {out_export.shape}")
    model.switch_to_train()
    print("  [OK]")

    # ── 测试 5: bias_init ──
    print("\n[Test 5] bias_init")
    model.bias_init()
    print("  [OK]")

    # ── 测试 6: isinstance(m, Detect) ──
    print("\n[Test 6] isinstance(m, Detect) 兼容性")
    assert isinstance(model, Detect), "isinstance(m, Detect) 必须为 True"
    print("  [OK] ← 训练引擎的 stride/device/fuse 全部自动兼容")

    # ── 测试 7: 梯度回传 ──
    print("\n[Test 7] 梯度回传 (训练模式)")
    model.train()
    model.switch_to_train()
    feats_grad = [f.requires_grad_(True) for f in feats]
    out_grad = model(feats_grad)
    # 主分支 + aux 分支联合 loss
    loss = (
        out_grad["boxes"].sum()
        + out_grad["scores"].sum()
        + out_grad["aux"]["boxes"].sum()
    )
    loss.backward()
    grad_ok = all(
        f.grad is not None and f.grad.abs().sum().item() > 0
        for f in feats_grad
    )
    assert grad_ok, "梯度回传异常"
    print(f"  梯度: {[f.grad.abs().sum().item() for f in feats_grad]}")
    print("  [OK]")

    # ── 测试 8: 训练模式下 export=True 阻断 P2 ──
    print("\n[Test 8] 训练 + export=True (P2 阻断)")
    model.train()
    model.switch_to_export()
    out_export_train = model(feats)
    # export=True 时即使 training=True 也不返回 aux
    assert "aux" not in out_export_train, "export 模式不应有 aux"
    print("  [OK] export 模式下 P2 已阻断")

    # ── 参数统计 ──
    total_params = sum(p.numel() for p in model.parameters())
    main_params = sum(p.numel() for p in model.cv2.parameters()) + sum(
        p.numel() for p in model.cv3.parameters()
    )
    aux_params = sum(p.numel() for p in model.aux_cv2.parameters()) + sum(
        p.numel() for p in model.aux_cv3.parameters()
    )
    dfl_params = sum(p.numel() for p in model.dfl.parameters())
    print(f"\n[PGI_Detect 参数统计]:")
    print(f"  总参数量:     {total_params:,}")
    print(f"  主分支:       {main_params:,}")
    print(f"  P2 辅助分支:  {aux_params:,} ({100*aux_params/total_params:.1f}%)")
    print(f"  DFL:          {dfl_params:,}")
    print(f"  推理时 P2 剥离: 零额外算力")

    print(f"\n{'=' * 60}")
    print("[PASS] PGI_Detect 全部兼容性测试通过!")
    print(f"{'=' * 60}")
