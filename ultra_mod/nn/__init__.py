# 自定义模块初始化
from .modules import SCABlock, GSConv, GSC_Bottleneck_Cross, C2f_GSC_Cross, PGI_Detect
from .losses import wasserstein_distance_loss, BboxLossWithNWD

__all__ = (
    "SCABlock",
    "GSConv",
    "GSC_Bottleneck_Cross",
    "C2f_GSC_Cross",
    "PGI_Detect",
    "BboxLossWithNWD",
    "wasserstein_distance_loss",
)
