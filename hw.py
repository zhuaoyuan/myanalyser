import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Callable, Optional

import akshare as ak
import pandas as pd
from pybroker.common import DataCol, to_datetime
from pybroker.data import DataSource

BAR_BASE_COLS = [
    DataCol.DATE.value,
    DataCol.SYMBOL.value,
    DataCol.OPEN.value,
    DataCol.HIGH.value,
    DataCol.LOW.value,
    DataCol.CLOSE.value,
    DataCol.VOLUME.value,
]
BAR_EXTRA_COLS = [
    "amount",
    "amplitude",
    "change_pct",
    "change",
    "turnover",
    "nav_accum",
    "nav_daily_pct",
    "adjust",
    "price_source",
    "volume_source",
]
BAR_COLS = BAR_BASE_COLS + BAR_EXTRA_COLS


@dataclass
class FetchError:
    symbol: str
    reason: str


@dataclass
class ActionError:
    symbol: str
    reason: str


def clear_proxy_env():
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"


def is_ac_share_class(name: str) -> bool:
    normalized = str(name).replace(" ", "").upper()
    if any(token in normalized for token in ("A类".upper(), "C类".upper())):
        return True
    return normalized.endswith("A") or normalized.endswith("C")


def normalize_adjust(adjust: Optional[str]) -> str:
    value = (adjust or "").strip().lower()
    if value not in {"", "qfq", "hfq"}:
        raise ValueError("adjust must be one of: '', qfq, hfq")
    return value


def debug_log(debug: bool, message: str) -> None:
    if debug:
        print(f"[fund_universe] {message}")


def normalize_overview_columns(df: pd.DataFrame) -> pd.DataFrame:
    targets = [
        "基金管理人",
        "基金经理人",
        "发行日期",
        "净资产规模",
        "管理费率",
        "托管费率",
        "销售服务费率",
        "最高认购费率",
        "最高申购费率",
        "最高赎回费率",
    ]
    if "基金代码" in df.columns:
        df = df.copy()
        df["基金代码"] = df["基金代码"].astype(str).str.extract(r"(\\d{6})")[0]
    if df.shape[1] == 2 and not any(col in df.columns for col in targets):
        key_col, val_col = df.columns.tolist()
        long_map = df.set_index(key_col)[val_col].to_dict()
        df = pd.DataFrame([long_map])
        if "基金代码" in df.columns:
            df["基金代码"] = df["基金代码"].astype(str).str.extract(r"(\\d{6})")[0]
    candidates = {
        "基金管理人": ["基金管理人", "基金公司", "基金公司名称"],
        "基金经理人": ["基金经理人", "基金经理", "基金经理(现任)", "基金经理(在任)"],
        "发行日期": ["发行日期", "成立日期", "成立时间", "成立日"],
        "净资产规模": ["净资产规模", "最新规模", "基金规模", "最新规模(亿元)", "规模(亿元)"],
        "管理费率": ["管理费率", "管理费", "管理费率(%)"],
        "托管费率": ["托管费率", "托管费", "托管费率(%)"],
        "销售服务费率": ["销售服务费率", "销售服务费", "销售服务费率(%)"],
        "最高认购费率": ["最高认购费率", "最高认购费率(%)", "认购费率上限", "认购费率(最高)"],
        "最高申购费率": ["最高申购费率", "最高申购费率(%)", "申购费率上限", "申购费率(最高)"],
        "最高赎回费率": ["最高赎回费率", "最高赎回费率(%)", "赎回费率上限", "赎回费率(最高)"],
    }
    rename_map: dict[str, str] = {}
    for target in targets:
        for col in candidates[target]:
            if col in df.columns:
                rename_map[col] = target
                break
    out = df.rename(columns=rename_map).copy()
    for target in targets:
        if target not in out.columns:
            out[target] = pd.NA
    return out[["基金代码"] + targets]


def fetch_overview_for_codes(codes: list[str], debug: bool = False) -> pd.DataFrame:

    debug_log(debug, f"codes: {codes}")

    frames: list[pd.DataFrame] = []
    for idx_code, code in enumerate(codes):
        last_err: Optional[Exception] = None
        for idx in range(3):
            try:
                df = ak.fund_overview_em(symbol=code)

                if df is None or df.empty:
                    if debug and idx_code < 3:
                        debug_log(debug, f"overview(symbol={code}) empty response")
                    break
                df = df.copy()
                df["基金代码"] = code
                frames.append(df)
                if debug and idx_code < 3:
                    debug_log(
                        debug, f"overview(symbol={code}) columns: {df.columns.tolist()}"
                    )
                    debug_log(debug, f"overview(symbol={code}) sample:\n{df.head(5)}")
                break
            except Exception as err:  # noqa: BLE001
                last_err = err
                if debug and idx_code < 3:
                    debug_log(debug, f"overview(symbol={code}) error: {err}")
                if idx < 2:
                    time.sleep(0.6 * (idx + 1))
        if last_err is not None and not frames:
            continue
    if not frames:
        return pd.DataFrame(columns=["基金代码"])
    return pd.concat(frames, ignore_index=True)


def append_fund_overview(universe: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    if universe.empty or "基金代码" not in universe.columns:
        return universe
    overview_raw = fetch_overview_for_codes(universe["基金代码"].tolist(), debug=debug)
    if debug:
        debug_log(debug, f"overview raw columns: {overview_raw.columns.tolist()}")
        debug_log(debug, f"overview raw sample:\n{overview_raw.head(3)}")
    # overview = normalize_overview_columns(overview_raw)
    # if debug:
    #     missing_stats = overview.isna().mean().to_dict()
    #     debug_log(debug, f"overview missing ratios: {missing_stats}")
    #     debug_log(debug, f"overview normalized sample:\n{overview.head(3)}")
    if not overview_raw.empty:
        return universe.merge(overview_raw, how="left", on="基金代码")
    return universe


def get_fund_universe(debug: bool = False, include_overview: bool = True) -> pd.DataFrame:
    def normalize_purchase_columns(df: pd.DataFrame) -> pd.DataFrame:
        base_cols = ["基金代码", "基金简称", "基金类型", "申购状态", "赎回状态"]
        rename_map: dict[str, str] = {}
        candidates = {
            "日累计限定金额": [
                "日累计限定金额",
                "日累计限额",
                "日累计限额(元)",
                "日累计限定金额(元)",
            ],
            "手续费": ["手续费", "手续费率", "手续费(%)", "申购手续费"],
        }
        for target, options in candidates.items():
            for col in options:
                if col in df.columns:
                    rename_map[col] = target
                    break
        out = df.rename(columns=rename_map).copy()
        for target in candidates:
            if target not in out.columns:
                out[target] = pd.NA
        cols = [c for c in base_cols if c in out.columns] + list(candidates.keys())
        return out[cols]

    fund_purchase_raw = ak.fund_purchase_em().copy()
    debug_log(debug, f"purchase columns: {fund_purchase_raw.columns.tolist()}")
    fund_purchase_raw["基金代码"] = fund_purchase_raw["基金代码"].astype(str).str.zfill(6)
    fund_purchase = normalize_purchase_columns(fund_purchase_raw)
    if debug:
        debug_log(debug, f"purchase sample:\n{fund_purchase.head(3)}")

    otc = fund_purchase[
        fund_purchase["申购状态"].astype(str).str.contains("开放申购", na=False)
    ].copy()
    otc = otc[otc["基金简称"].map(is_ac_share_class)].copy()
    otc = otc.drop_duplicates(subset=["基金代码"])
    otc["market"] = "otc_open_ac"
    otc["symbol"] = "OF." + otc["基金代码"]

    exchange = ak.fund_etf_spot_ths()[
        ["基金代码", "基金名称", "基金类型", "申购状态", "赎回状态"]
    ].copy()
    exchange = exchange.rename(columns={"基金名称": "基金简称"})
    exchange["基金代码"] = exchange["基金代码"].astype(str).str.zfill(6)
    exchange = exchange.drop_duplicates(subset=["基金代码"])
    exchange["market"] = "exchange_fund"
    exchange["symbol"] = "EX." + exchange["基金代码"]

    cols = [
        "symbol",
        "基金代码",
        "基金简称",
        "基金类型",
        "申购状态",
        "赎回状态",
        "日累计限定金额",
        "手续费",
        "market",
    ]
    cols_available = [col for col in cols if col in otc.columns]
    otc = otc[cols_available].copy()

    universe = pd.concat([otc, exchange], ignore_index=True)
    universe = universe.drop_duplicates(subset=["symbol"])
    if include_overview:
        universe = append_fund_overview(universe, debug=debug)
    return universe.reset_index(drop=True)


def pick_representatives(
    universe: pd.DataFrame,
    per_market: int = 0,
    per_type: int = 0,
    limit: int = 0,
) -> pd.DataFrame:
    if universe.empty:
        return universe
    selected: list[int] = []
    if per_market > 0:
        for idx in universe.groupby("market", sort=False).head(per_market).index.tolist():
            if idx not in selected:
                selected.append(idx)
    if per_type > 0:
        for idx in universe.groupby("基金类型", sort=False).head(per_type).index.tolist():
            if idx not in selected:
                selected.append(idx)
    if selected:
        sampled = universe.loc[selected]
    else:
        sampled = universe

    if limit > 0:
        if len(sampled) < limit:
            rest = universe.drop(index=sampled.index).head(limit - len(sampled))
            sampled = pd.concat([sampled, rest], ignore_index=False)
        sampled = sampled.head(limit)
    return sampled.reset_index(drop=True)


def normalize_action_table(df: pd.DataFrame, symbol: str, event_type: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["symbol", "event_type", "event_date", "raw"])

    date_col = None
    for col in df.columns:
        col_s = str(col)
        if "日期" in col_s or "登记日" in col_s or "除息日" in col_s:
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]

    out = df.copy()
    out["event_date"] = pd.to_datetime(out[date_col], errors="coerce")
    out["symbol"] = symbol
    out["event_type"] = event_type
    out["raw"] = out.apply(
        lambda row: json.dumps(row.to_dict(), ensure_ascii=False, default=str), axis=1
    )
    return out[["symbol", "event_type", "event_date", "raw"]]


def fetch_symbol_actions(symbol: str, code: str, market: str) -> pd.DataFrame:
    frames = []
    if market == "OF":
        div_df = ak.fund_open_fund_info_em(symbol=code, indicator="分红送配详情")
        split_df = ak.fund_open_fund_info_em(symbol=code, indicator="拆分详情")
        frames.append(normalize_action_table(div_df, symbol, "dividend"))
        frames.append(normalize_action_table(split_df, symbol, "split"))
    elif market == "EX":
        ex_symbol = f"{'sh' if code.startswith(('5', '6')) else 'sz'}{code}"
        div_df = ak.fund_etf_dividend_sina(symbol=ex_symbol)
        if not div_df.empty:
            div_df = div_df.rename(columns={"日期": "event_date", "累计分红": "cumulative_dividend"})
            div_df["event_date"] = pd.to_datetime(div_df["event_date"], errors="coerce")
            div_df["symbol"] = symbol
            div_df["event_type"] = "dividend_cum"
            div_df["raw"] = div_df.apply(
                lambda row: json.dumps(row.to_dict(), ensure_ascii=False, default=str), axis=1
            )
            frames.append(div_df[["symbol", "event_type", "event_date", "raw"]])
    if not frames:
        return pd.DataFrame(columns=["symbol", "event_type", "event_date", "raw"])
    return pd.concat(frames, ignore_index=True)


def collect_corporate_actions(
    universe: pd.DataFrame,
    retries: int,
    retry_wait: float,
) -> tuple[pd.DataFrame, list[ActionError]]:
    errors: list[ActionError] = []
    frames = []

    def with_retry(fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        last_err: Optional[Exception] = None
        for idx in range(retries):
            try:
                return fn()
            except Exception as err:  # noqa: BLE001
                last_err = err
                if idx < retries - 1:
                    time.sleep(retry_wait * (idx + 1))
        assert last_err is not None
        raise last_err

    for sym in universe["symbol"]:
        market, code = sym.split(".", 1)
        try:
            actions = with_retry(lambda: fetch_symbol_actions(sym, code, market))
            if not actions.empty:
                frames.append(actions)
        except Exception as err:  # noqa: BLE001
            errors.append(ActionError(symbol=sym, reason=str(err)))

    if not frames:
        return pd.DataFrame(columns=["symbol", "event_type", "event_date", "raw"]), errors
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["event_date", "symbol", "event_type"]).reset_index(drop=True)
    return out, errors


class CNFundDataSource(DataSource):
    def __init__(self, max_workers: int = 1, retries: int = 3, retry_wait: float = 0.8):
        super().__init__()
        self.max_workers = max_workers
        self.retries = retries
        self.retry_wait = retry_wait
        self.errors: list[FetchError] = []
        self._lock = Lock()

    def _with_retry(self, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        last_err: Optional[Exception] = None
        for idx in range(self.retries):
            try:
                return fn()
            except Exception as err:  # noqa: BLE001
                last_err = err
                if idx < self.retries - 1:
                    time.sleep(self.retry_wait * (idx + 1))
        assert last_err is not None
        raise last_err

    def _empty_bar_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=BAR_COLS)

    def _fetch_otc(self, symbol: str, code: str, start_str: str, end_str: str, adjust: str) -> pd.DataFrame:
        unit = self._with_retry(
            lambda: ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        )
        if unit.empty:
            return self._empty_bar_frame()

        unit = unit.rename(
            columns={"净值日期": "date", "单位净值": "close", "日增长率": "nav_daily_pct"}
        )[["date", "close", "nav_daily_pct"]]
        unit["date"] = pd.to_datetime(unit["date"], errors="coerce")

        accum = self._with_retry(
            lambda: ak.fund_open_fund_info_em(symbol=code, indicator="累计净值走势")
        )
        if accum.empty:
            accum = pd.DataFrame(columns=["date", "nav_accum"])
        else:
            accum = accum.rename(columns={"净值日期": "date", "累计净值": "nav_accum"})[
                ["date", "nav_accum"]
            ]
            accum["date"] = pd.to_datetime(accum["date"], errors="coerce")

        raw = unit.merge(accum, how="left", on="date")
        raw = raw[(raw["date"] >= pd.to_datetime(start_str)) & (raw["date"] <= pd.to_datetime(end_str))]
        raw["symbol"] = symbol
        raw["open"] = pd.to_numeric(raw["close"], errors="coerce")
        raw["high"] = pd.to_numeric(raw["close"], errors="coerce")
        raw["low"] = pd.to_numeric(raw["close"], errors="coerce")
        raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
        raw["volume"] = pd.NA
        raw["amount"] = pd.NA
        raw["amplitude"] = pd.NA
        raw["change_pct"] = pd.to_numeric(raw["nav_daily_pct"], errors="coerce")
        raw["change"] = pd.NA
        raw["turnover"] = pd.NA
        raw["nav_accum"] = pd.to_numeric(raw["nav_accum"], errors="coerce")
        raw["nav_daily_pct"] = pd.to_numeric(raw["nav_daily_pct"], errors="coerce")
        raw["adjust"] = adjust
        raw["price_source"] = "eastmoney_open_fund"
        raw["volume_source"] = "not_applicable"
        return raw[BAR_COLS].dropna(subset=["date", "close"])

    def _fetch_exchange(self, symbol: str, code: str, adjust: str) -> pd.DataFrame:
        fetchers = [
            (
                "eastmoney_etf_em",
                lambda: ak.fund_etf_hist_em(
                    symbol=code,
                    period="daily",
                    start_date="19900101",
                    end_date="21000101",
                    adjust=adjust,
                ),
            ),
            (
                "eastmoney_lof_em",
                lambda: ak.fund_lof_hist_em(
                    symbol=code,
                    period="daily",
                    start_date="19900101",
                    end_date="21000101",
                    adjust=adjust,
                ),
            ),
        ]
        if adjust == "":
            fetchers.append(
                (
                    "sina_etf",
                    lambda: ak.fund_etf_hist_sina(
                        symbol=f"{'sh' if code.startswith(('5', '6')) else 'sz'}{code}"
                    ),
                )
            )

        raw = pd.DataFrame()
        errors = []
        source_name = ""
        for name, fetch_fn in fetchers:
            try:
                raw = self._with_retry(fetch_fn)
                if not raw.empty:
                    source_name = name
                    break
            except Exception as err:  # noqa: BLE001
                errors.append(str(err))
                continue
        if raw.empty:
            if errors:
                raise RuntimeError(" | ".join(errors))
            return self._empty_bar_frame()

        raw = raw.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "change_pct",
                "涨跌额": "change",
                "换手率": "turnover",
            }
        )
        raw["symbol"] = symbol
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        for col in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "amplitude",
            "change_pct",
            "change",
            "turnover",
        ):
            if col not in raw.columns:
                raw[col] = pd.NA
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw["nav_accum"] = pd.NA
        raw["nav_daily_pct"] = pd.NA
        raw["adjust"] = adjust
        raw["price_source"] = source_name
        raw["volume_source"] = "exchange_trade_volume"
        return raw[BAR_COLS].dropna(subset=["date", "close"])

    def _fetch_one(
        self, symbol: str, start_date: datetime, end_date: datetime, adjust: str
    ) -> pd.DataFrame:
        market, code = symbol.split(".", 1)
        start_str = to_datetime(start_date).strftime("%Y-%m-%d")
        end_str = to_datetime(end_date).strftime("%Y-%m-%d")
        if market == "OF":
            return self._fetch_otc(symbol, code, start_str, end_str, adjust)
        if market == "EX":
            df = self._fetch_exchange(symbol, code, adjust)
            if df.empty:
                return df
            return df[(df["date"] >= pd.to_datetime(start_str)) & (df["date"] <= pd.to_datetime(end_str))]
        return self._empty_bar_frame()

    def _fetch_data(
        self,
        symbols: frozenset[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: Optional[str],
        adjust: Optional[str],
    ) -> pd.DataFrame:
        if timeframe not in ("", "1day"):
            raise ValueError("CNFundDataSource only supports daily timeframe.")
        adjust_mode = normalize_adjust(adjust)
        all_frames = []
        if self.max_workers <= 1:
            for sym in symbols:
                try:
                    df = self._fetch_one(sym, start_date, end_date, adjust_mode)
                    if not df.empty:
                        all_frames.append(df)
                except Exception as err:  # noqa: BLE001
                    with self._lock:
                        self.errors.append(FetchError(symbol=sym, reason=str(err)))
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                tasks = {
                    executor.submit(self._fetch_one, sym, start_date, end_date, adjust_mode): sym
                    for sym in symbols
                }
                for future in as_completed(tasks):
                    sym = tasks[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            all_frames.append(df)
                    except Exception as err:  # noqa: BLE001
                        with self._lock:
                            self.errors.append(FetchError(symbol=sym, reason=str(err)))
        if not all_frames:
            return self._empty_bar_frame()
        return pd.concat(all_frames, ignore_index=True)


def main():
    clear_proxy_env()
    parser = argparse.ArgumentParser(
        description="Fetch China OTC+exchange fund daily bars and corporate actions via pybroker."
    )
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-12-31")
    parser.add_argument("--output-dir", default="data/cn_funds_2026")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=0, help="cap symbol count after sampling")
    parser.add_argument("--sample-per-market", type=int, default=0)
    parser.add_argument("--sample-per-type", type=int, default=0)
    parser.add_argument("--adjust", default="", choices=["", "qfq", "hfq"])
    parser.add_argument("--debug", action="store_true", help="print debug info for fund universe")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    universe = get_fund_universe(debug=args.debug, include_overview=False)
    universe = pick_representatives(
        universe,
        per_market=max(args.sample_per_market, 0),
        per_type=max(args.sample_per_type, 0),
        limit=max(args.limit, 0),
    )
    universe = append_fund_overview(universe, debug=args.debug)
    symbols = universe["symbol"].tolist()

    data_source = CNFundDataSource(
        max_workers=args.workers,
        retries=args.retries,
        retry_wait=args.retry_wait,
    )
    bars = data_source.query(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe="1day",
        adjust=args.adjust,
    )

    out = bars.merge(universe, how="left", on="symbol")
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)

    # actions_df, action_errors = collect_corporate_actions(
    #     universe=universe,
    #     retries=args.retries,
    #     retry_wait=args.retry_wait,
    # )

    universe_path = os.path.join(args.output_dir, "fund_universe.csv")
    bars_path = os.path.join(args.output_dir, "fund_daily_2026.parquet")
    errors_path = os.path.join(args.output_dir, "fetch_errors.csv")
    out_path_csv = os.path.join(args.output_dir, "fund_daily_2026.csv")
    actions_path = os.path.join(args.output_dir, "fund_corporate_actions_2026.csv")

    universe.to_csv(universe_path, index=False)
    parquet_saved = True
    try:
        out.to_parquet(bars_path, index=False)
    except Exception:  # noqa: BLE001
        parquet_saved = False
    out.to_csv(out_path_csv, index=False)
    # actions_df.to_csv(actions_path, index=False)

    all_errors = [e.__dict__ for e in data_source.errors]
    # all_errors.extend({"symbol": e.symbol, "reason": f"actions: {e.reason}"} for e in action_errors)
    pd.DataFrame(all_errors, columns=["symbol", "reason"]).to_csv(errors_path, index=False)

    print(f"symbols_total={len(symbols)}")
    print(f"symbols_failed={len(data_source.errors)}")
    # print(f"action_symbols_failed={len(action_errors)}")
    print(f"rows={len(out)}")
    # print(f"actions_rows={len(actions_df)}")
    print(f"saved_universe={universe_path}")
    print(f"saved_parquet={bars_path}" if parquet_saved else "saved_parquet=skipped")
    print(f"saved_csv={out_path_csv}")
    print(f"saved_actions={actions_path}")
    print(f"saved_errors={errors_path}")


if __name__ == "__main__":
    main()
