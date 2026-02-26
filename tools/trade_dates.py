import akshare as ak
from pathlib import Path


tool_trade_date_hist_sina_df = ak.tool_trade_date_hist_sina()
project_root = Path(__file__).resolve().parent.parent
output_csv = project_root / "data" / "common" / "trade_dates.csv"
output_csv.parent.mkdir(parents=True, exist_ok=True)
tool_trade_date_hist_sina_df.to_csv(output_csv, index=False)
print(f"saved: {output_csv}")