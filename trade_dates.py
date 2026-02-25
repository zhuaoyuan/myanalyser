import akshare as ak
tool_trade_date_hist_sina_df = ak.tool_trade_date_hist_sina()
tool_trade_date_hist_sina_df.to_csv("trade_dates.csv", index=False)