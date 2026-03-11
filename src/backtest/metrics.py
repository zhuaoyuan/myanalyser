"""回测所需指标计算（复用 fund_metrics_core）。"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from fund_metrics_core import (
        WindowConfig,
        compute_low_risk_debt_metrics,
    )
except ModuleNotFoundError:
    # CLI/非 pytest 直接运行时 src 可能不在 path，兜底注入
    _src = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_src))
    from fund_metrics_core import (
        WindowConfig,
        compute_low_risk_debt_metrics,
    )

__all__ = ["WindowConfig", "compute_low_risk_debt_metrics"]
