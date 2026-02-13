import akshare as ak
res = ak.fund_overview_em(symbol="000001")
print(res.columns.tolist())
print(res.head())