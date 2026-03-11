"""回测所需指标计算（复用 fund_metrics_core）。"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保 fund_metrics_core 可被导入（pytest 等可能未将 src 加入 path）
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from fund_metrics_core import (
    WindowConfig,
    compute_low_risk_debt_metrics,
)

__all__ = ["WindowConfig", "compute_low_risk_debt_metrics"]
