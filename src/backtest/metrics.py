"""回测所需指标计算（复用 fund_metrics_core）。"""

from __future__ import annotations

from fund_metrics_core import (
    WindowConfig,
    compute_low_risk_debt_metrics,
)

__all__ = ["WindowConfig", "compute_low_risk_debt_metrics"]
