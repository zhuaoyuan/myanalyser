# import akshare as ak
# res = ak.fund_overview_em(symbol="000001")
# print(res.columns.tolist())
# print(res.head())


import akshare as ak

# fund_lof_hist_em_df = ak.fund_lof_hist_em(symbol="166009", period="daily", start_date="20200101", end_date="20260220", adjust="hfq")
# print(fund_lof_hist_em_df)


fund_open_fund_info_em_df = ak.fund_open_fund_info_em(symbol="000009", indicator="累计净值走势")
print(fund_open_fund_info_em_df)

# import akshare as ak

# fund_purchase_em_df = ak.fund_purchase_em()
# print(fund_purchase_em_df)

# fund_name_em_df = ak.fund_name_em()
# print(fund_name_em_df)

# fund_purchase_em_df = ak.fund_purchase_em()
# print(fund_purchase_em_df)
