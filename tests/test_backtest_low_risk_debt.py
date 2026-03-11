"""低风险偏债策略在回测框架中的评分测试。"""

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

_ws = Path(__file__).resolve().parents[2]
if str(_ws) not in sys.path:
    sys.path.insert(0, str(_ws))

from myanalyser.src.backtest.data import BacktestData
from myanalyser.src.backtest.strategies.low_risk_debt import LowRiskDebtScoreStrategy


class LowRiskDebtScoreStrategyTest(unittest.TestCase):
    def test_score_prefers_stronger_trend(self) -> None:
        dates = pd.date_range("2020-01-01", periods=800, freq="B")
        nav_up = 1.0 + np.linspace(0, 1.0, len(dates))
        nav_flat = 1.0 + np.linspace(0, 0.05, len(dates))

        df_up = pd.DataFrame({"date": dates, "close": nav_up})
        df_flat = pd.DataFrame({"date": dates, "close": nav_flat})

        long_rows = []
        for symbol, df in [("000001", df_up), ("000002", df_flat)]:
            for _, r in df.iterrows():
                long_rows.append(
                    {
                        "symbol": symbol,
                        "date": r["date"],
                        "open": r["close"],
                        "high": r["close"],
                        "low": r["close"],
                        "close": r["close"],
                    }
                )

        long_df = pd.DataFrame(long_rows)
        data = BacktestData(
            long_df=long_df,
            by_symbol={"000001": df_up, "000002": df_flat},
            trading_dates=list(dates),
        )

        strategy = LowRiskDebtScoreStrategy()
        scored = strategy.score(data, dates[-1], ["000001", "000002"])

        self.assertFalse(scored.empty)
        self.assertIn("综合得分", scored.columns)
        self.assertIn("综合排名", scored.columns)

        top_symbol = scored.iloc[0]["symbol"]
        self.assertEqual(top_symbol, "000001")


if __name__ == "__main__":
    unittest.main()
