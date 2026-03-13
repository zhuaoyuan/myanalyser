#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基金规模变动数据抓取模块（东方财富 FundArchivesDatas API）

提供 akshare 风格接口：输入基金编码，返回包含
日期、期间申购（亿份）、期间赎回（亿份）、期末总份额（亿份）、期末净资产（亿元）、净资产变动率
的 pandas DataFrame。

数据来源：https://fundf10.eastmoney.com/FundArchivesDatas.aspx?code={code}&type=gmbd
"""
from __future__ import annotations

import json
import logging
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# 东方财富请求头，降低被限流/拒绝概率
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://fundf10.eastmoney.com/",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def _normalize_code(code: str) -> str:
    raw = str(code).strip()
    if raw.isdigit():
        return raw.zfill(6)
    return raw


def _to_yiyuan(val) -> float | None:
    """原始值为元，转为亿元。空/--- 返回 None。"""
    if val is None or val == "" or val == "---":
        return None
    try:
        v = float(val)
        return round(v / 1e8, 2) if v != 0 else 0.0
    except (ValueError, TypeError):
        return None


def _to_pct(val) -> str | None:
    """净资产变动率：原始为小数如 -3.39，转为 -3.39%。"""
    if val is None or val == "":
        return None
    try:
        v = float(val)
        return f"{v:.2f}%"
    except (ValueError, TypeError):
        return str(val) if val else None


def _parse_num(val: str) -> float | None:
    """解析数值：去除千分位逗号，--- 返回 None。"""
    if val is None or not val or str(val).strip() == "---":
        return None
    s = str(val).strip().replace(",", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


_EMPTY_COLUMNS = [
    "日期", "期间申购（亿份）", "期间赎回（亿份）",
    "期末总份额（亿份）", "期末净资产（亿元）", "净资产变动率"
]


def fetch_gmbd_api(
    code: str,
    page: int = 1,
    per: int = 200,
    timeout: int = 15,
    headers: dict | None = None,
) -> dict:
    """
    请求 FundArchivesDatas API，返回 gmbd_apidata 对象。

    参数
    -----
    code : str
        基金编码
    page : int
        页码，默认 1
    per : int
        每页条数，默认 200
    timeout : int
        请求超时秒数
    headers : dict, optional
        自定义请求头，默认使用 DEFAULT_HEADERS

    返回
    -----
    dict
        {"content": str, "summary": str, "data": list}
    """
    code = _normalize_code(code)
    if not code or not code.isdigit():
        return {"content": "", "summary": "", "data": []}
    url = "https://fundf10.eastmoney.com/FundArchivesDatas.aspx"
    params = {"code": code, "type": "gmbd", "page": page, "per": per}
    hdrs = headers or DEFAULT_HEADERS
    resp = requests.get(url, params=params, headers=hdrs, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    text = resp.text

    # 解析 JS 返回值：var gmbd_apidata={ content:"<table>...</table>", summary:"...", data:[...]};
    # data 数组为扁平对象列表，无嵌套数组，非贪婪匹配可正确截断
    content_m = re.search(r'content\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    content = content_m.group(1) if content_m else ""

    data: list = []
    data_m = re.search(r',\s*data\s*:\s*(\[[\s\S]*?\])\s*}', text)
    if data_m:
        try:
            data = json.loads(data_m.group(1))
        except json.JSONDecodeError:
            pass

    return {"content": content, "summary": "", "data": data}


def _parse_gmbd_from_content(html: str) -> pd.DataFrame:
    """从 content 内的 HTML 表格解析为 DataFrame。"""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="gmbd") or soup.find("table")
    if not table:
        return pd.DataFrame()

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) == len(headers):
            rows.append(cells)
    if not headers or not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=headers)

    num_cols = ["期间申购（亿份）", "期间赎回（亿份）", "期末总份额（亿份）", "期末净资产（亿元）"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].apply(_parse_num)
    return df


def fund_gmbd_em(code: str, timeout: int = 15) -> pd.DataFrame:
    """
    获取基金规模变动数据（akshare 风格 API）

    参数
    -----
    code : str
        基金编码，如 000198
    timeout : int
        请求超时秒数，默认 15

    返回
    -----
    pd.DataFrame
        列：日期、期间申购（亿份）、期间赎回（亿份）、期末总份额（亿份）、期末净资产（亿元）、净资产变动率
        数值列已转为 float，--- 为 NaN；按日期倒序（最新在前）
    """
    code = _normalize_code(code)
    if not code or not code.isdigit():
        return pd.DataFrame(columns=_EMPTY_COLUMNS)
    apidata = fetch_gmbd_api(code, timeout=timeout)
    rows = apidata.get("data") or []
    content = apidata.get("content") or ""

    if rows:
        records = []
        for r in rows:
            dt = r.get("FSRQ") or ""
            sg = _to_yiyuan(r.get("QJSG"))
            sh = _to_yiyuan(r.get("QJSH"))
            mjzc = _to_yiyuan(r.get("QMJZC"))  # 期末总份额（亿份）
            mzfe = _to_yiyuan(r.get("QMZFE"))  # 期末净资产（亿元）
            # 货币基金等品种 API 有时仅返回 QMZFE，QMJZC 为空；份额=净资产，故 fallback
            if mjzc is None and mzfe is not None:
                mjzc = mzfe
            zfebdl = _to_pct(r.get("ZFEBDL")) or r.get("CHANGE")
            records.append({
                "日期": dt,
                "期间申购（亿份）": sg,
                "期间赎回（亿份）": sh,
                "期末总份额（亿份）": mjzc,
                "期末净资产（亿元）": mzfe,
                "净资产变动率": zfebdl,
            })
        return pd.DataFrame(records)

    if content:
        df = _parse_gmbd_from_content(content)
        if not df.empty:
            return df

    return pd.DataFrame(columns=_EMPTY_COLUMNS)
