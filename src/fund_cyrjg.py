#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基金持有人结构数据抓取模块（东方财富 FundArchivesDatas API）

提供 akshare 风格接口：输入基金编码，返回包含
日期、机构持有比例、个人持有比例、内部持有比例、总份额（亿份）
的 pandas DataFrame。

数据来源：https://fundf10.eastmoney.com/FundArchivesDatas.aspx?code={code}&type=cyrjg
"""
from __future__ import annotations

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

# 东方财富请求头，降低被限流/拒绝概率
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://fundf10.eastmoney.com/",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

_EMPTY_COLUMNS = [
    "日期",
    "机构持有比例",
    "个人持有比例",
    "内部持有比例",
    "总份额（亿份）",
]


def _normalize_code(code: str) -> str:
    raw = str(code).strip()
    if raw.isdigit():
        return raw.zfill(6)
    return raw


def _parse_num(val: str) -> float | None:
    """解析数值：去除千分位逗号，--- 返回 None。"""
    if val is None or not val or str(val).strip() == "---":
        return None
    s = str(val).strip().replace(",", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def fetch_cyrjg_api(
    code: str,
    page: int = 1,
    per: int = 200,
    timeout: int = 15,
    headers: dict | None = None,
) -> dict:
    """
    请求 FundArchivesDatas API（type=cyrjg），返回 apidata 对象。

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
        {"content": str, "summary": str}
        content 为 Markdown 表格
    """
    code = _normalize_code(code)
    if not code or not code.isdigit():
        return {"content": "", "summary": ""}
    url = "https://fundf10.eastmoney.com/FundArchivesDatas.aspx"
    params = {"code": code, "type": "cyrjg", "page": page, "per": per}
    hdrs = headers or DEFAULT_HEADERS
    resp = requests.get(url, params=params, headers=hdrs, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    text = resp.text

    # 解析 var apidata={ content:"...", summary:"..." };
    content_m = re.search(r'content\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    content = content_m.group(1) if content_m else ""

    summary_m = re.search(r'summary\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    summary = summary_m.group(1) if summary_m else ""

    return {"content": content, "summary": summary}


def _parse_cyrjg_from_html(html: str) -> pd.DataFrame:
    """从 HTML 表格解析为 DataFrame。API 实际返回 HTML 而非 Markdown。"""
    if not html or not html.strip():
        return pd.DataFrame()
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_=re.compile(r"cyrjg", re.I)) or soup.find("table")
    if not table:
        return pd.DataFrame()
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    if not headers:
        return pd.DataFrame()
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) == len(headers):
            rows.append(cells)
    if not headers or not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=headers)
    if "公告日期" in df.columns:
        df = df.rename(columns={"公告日期": "日期"})
    if "总份额（亿份）" in df.columns:
        df["总份额（亿份）"] = df["总份额（亿份）"].apply(_parse_num)
    return df


def _parse_cyrjg_from_content(content: str) -> pd.DataFrame:
    """从 content 解析 DataFrame：优先 HTML 表格，否则尝试 Markdown 表格。"""
    if not content or not content.strip():
        return pd.DataFrame()
    df = _parse_cyrjg_from_html(content)
    if not df.empty:
        return df
    return _parse_cyrjg_from_markdown(content)


def _parse_cyrjg_from_markdown(content: str) -> pd.DataFrame:
    """从 Markdown 表格解析为 DataFrame（备用，API 可能返回此格式）。"""
    if not content or not content.strip():
        return pd.DataFrame()

    lines = [ln.strip() for ln in content.strip().split("\n") if ln.strip()]
    if len(lines) < 2:
        return pd.DataFrame()

    # 第一行：表头 | 公告日期 | 机构持有比例 | ...
    header_line = lines[0]
    if not header_line.startswith("|") or not header_line.endswith("|"):
        return pd.DataFrame()
    headers = [c.strip() for c in header_line.split("|")[1:-1]]
    if not headers:
        return pd.DataFrame()

    # 第二行：分隔符 | --- | --- | ...
    if len(lines) < 3:
        return pd.DataFrame()

    rows = []
    for line in lines[2:]:
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) == len(headers):
            rows.append(cells)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=headers)

    # 公告日期 -> 日期（与 gmbd 统一）
    if "公告日期" in df.columns:
        df = df.rename(columns={"公告日期": "日期"})

    # 比例列保留原样（如 63.45%）
    # 总份额（亿份）转为 float
    if "总份额（亿份）" in df.columns:
        df["总份额（亿份）"] = df["总份额（亿份）"].apply(_parse_num)

    return df


def fund_cyrjg_em(code: str, timeout: int = 15) -> pd.DataFrame:
    """
    获取基金持有人结构数据（akshare 风格 API）

    参数
    -----
    code : str
        基金编码，如 000015
    timeout : int
        请求超时秒数，默认 15

    返回
    -----
    pd.DataFrame
        列：日期、机构持有比例、个人持有比例、内部持有比例、总份额（亿份）
        比例列保留 % 字符串；总份额为 float；按日期倒序（最新在前）
    """
    code = _normalize_code(code)
    if not code or not code.isdigit():
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    apidata = fetch_cyrjg_api(code, timeout=timeout)
    content = apidata.get("content") or ""

    if content:
        df = _parse_cyrjg_from_content(content)
        if not df.empty:
            return df

    return pd.DataFrame(columns=_EMPTY_COLUMNS)
