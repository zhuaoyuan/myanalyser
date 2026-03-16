"""v2: 预备 eligible 按窗口计算（仅使用本地缓存，不联网）。

缓存机制：
- eligible_base_{start}_{end}.csv：c.1+a+b+e 结果，不依赖 personnel_dir
- personnel_excluded_{start}_{end}.csv：规则 f 排除的基金编码
- eligible_fund_candidates.csv：base - personnel_excluded，加载时合并
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from project_paths import project_root

_TRANSFORMS_DIR = Path(__file__).resolve().parent

_MYANALYSER = project_root()

_PREP_TOOLS = _MYANALYSER / "tools" / "prep"
if str(_MYANALYSER / "src") not in sys.path:
    sys.path.insert(0, str(_MYANALYSER / "src"))
if str(_PREP_TOOLS) not in sys.path:
    sys.path.insert(0, str(_PREP_TOOLS))

_MAX_PERSONNEL_WORKERS = 16


def _safe_code(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, int):
        return f"{v:06d}"
    if isinstance(v, float) and v.is_integer():
        return f"{int(v):06d}"
    s = str(v).strip()
    if not s or s == "---":
        return ""
    if not s.isdigit():
        return ""
    if len(s) > 6:
        return ""
    return s.zfill(6)


def _parse_date(text: object) -> pd.Timestamp | None:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    s = str(text).strip()
    if not s or s == "---":
        return None
    zh = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", s)
    if zh:
        y, m, d = zh.groups()
        return pd.to_datetime(f"{y}-{int(m):02d}-{int(d):02d}", errors="coerce")
    num = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", s)
    if num:
        return pd.to_datetime(num.group(0), errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _parse_pct(val: object) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s == "---":
        return None
    m = re.match(r"([\d.]+)\s*%", s)
    return float(m[1]) if m else None


def _has_consecutive_over_60(grp: pd.DataFrame, date_col: str) -> bool:
    df = grp.dropna(subset=["_pct", date_col]).sort_values(date_col)
    if len(df) < 2:
        return False
    df = df.groupby(date_col, as_index=False)["_pct"].max().sort_values(date_col)
    vals = df["_pct"].values
    for i in range(len(vals) - 1):
        if vals[i] > 60 and vals[i + 1] > 60:
            return True
    return False


def _check_personnel_one(
    path: Path,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> bool:
    """单文件判定：是否在 [window_start, window_end] 内有人事变动。供并发调用。"""
    try:
        df = pd.read_csv(path, dtype={"基金代码": str}, encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
        logging.getLogger(__name__).debug("[eligible] f 解析人事文件 %s 失败: %s", path, e)
        return False
    date_col = "公告日期" if "公告日期" in df.columns else "日期"
    if date_col not in df.columns or df.empty:
        return False
    df = df.copy()
    df["_dt"] = df[date_col].map(_parse_date)
    df = df.dropna(subset=["_dt"])
    if df.empty:
        return False
    return ((df["_dt"] >= window_start) & (df["_dt"] <= window_end)).any()


def _compute_personnel_excluded(
    codes: set[str],
    personnel_dir: Path,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    max_workers: int = _MAX_PERSONNEL_WORKERS,
) -> set[str]:
    """仅对 personnel_dir 中存在的 {code}.csv 做并发读取，返回窗口内有人事变动的基金编码集合。"""
    existent = [(c, personnel_dir / f"{c}.csv") for c in codes if (personnel_dir / f"{c}.csv").exists()]
    if not existent:
        return set()
    exclude_f: set[str] = set()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_code = {
            ex.submit(_check_personnel_one, p, window_start, window_end): c
            for c, p in existent
        }
        for future in as_completed(future_to_code):
            code = future_to_code[future]
            try:
                if future.result():
                    exclude_f.add(code)
            except Exception:  # noqa: BLE001
                pass
    return exclude_f


def _ensure_fee_filtered(
    fee_structured_csv: Path,
    fee_filtered_csv: Path,
    logger: logging.Logger,
) -> Path:
    if fee_filtered_csv.exists():
        return fee_filtered_csv
    if not fee_structured_csv.exists():
        raise FileNotFoundError(f"missing fee structured csv: {fee_structured_csv}")

    script_path = _PREP_TOOLS / "filter_fund_fee_by_holding.py"
    if not script_path.exists():
        raise FileNotFoundError(f"missing fee filter script: {script_path}")
    spec = importlib.util.spec_from_file_location("fee_filter", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load fee filter module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info("[c.1] 生成费率分类: %s -> %s", fee_structured_csv, fee_filtered_csv)
    module.run(fee_structured_csv, fee_filtered_csv)
    if not fee_filtered_csv.exists():
        raise RuntimeError("failed to generate fund_fee_filtered.csv")
    return fee_filtered_csv


def run(
    work_dir: Path,
    start_date: str,
    end_date: str,
    *,
    personnel_dir: Path | None = None,
    output_path: Path | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    log = logger or logging.getLogger(__name__)
    work_dir = work_dir.resolve()

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    if start_ts > end_ts:
        raise ValueError(f"start-date cannot be after end-date: {start_date} > {end_date}")

    purchase_csv = work_dir / "fund_purchase.csv"
    cyrjg_csv = work_dir / "fund_cyrjg.csv"
    gmbd_csv = work_dir / "fund_gmbd.csv"
    fee_structured_csv = work_dir / "fund_fee_structured.csv"
    fee_filtered_csv = work_dir / "fund_fee_filtered.csv"
    overview_csv = work_dir / "fund_overview.csv"

    for p in [purchase_csv, cyrjg_csv, gmbd_csv, fee_structured_csv, overview_csv]:
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    _ensure_fee_filtered(fee_structured_csv, fee_filtered_csv, log)

    purchase_df = pd.read_csv(purchase_csv, dtype=str, encoding="utf-8-sig")
    codes = {c for c in purchase_df["基金代码"].dropna().map(_safe_code).tolist() if c}
    log.info("[eligible] 原始候选 %d 只", len(codes))

    # c.1: 必须在 fee 分类结果中存在
    c1_df = pd.read_csv(fee_filtered_csv, dtype=str, encoding="utf-8-sig")
    code_col = "基金编码" if "基金编码" in c1_df.columns else "基金代码"
    c1_codes = {c for c in c1_df[code_col].dropna().map(_safe_code).tolist() if c}
    codes &= c1_codes
    log.info("[eligible] c.1 后 %d", len(codes))

    # 统一读取 overview，供 a、e 共用
    inc_by_code: dict[str, pd.Timestamp] = {}
    include_e: set[str] | None = None
    e_df = pd.read_csv(overview_csv, dtype=str, encoding="utf-8-sig")
    col = "成立日期/规模" if "成立日期/规模" in e_df.columns else "成立日期"
    if col in e_df.columns:
        e_df = e_df.copy()
        e_df["_code"] = e_df["基金代码"].map(_safe_code)
        e_df["_inc"] = e_df[col].map(_parse_date)
        inc_ok = e_df[e_df["_inc"].notna() & (e_df["_code"] != "")]
        inc_ok = inc_ok.sort_values("_inc", na_position="last")
        inc_dedup = inc_ok.drop_duplicates("_code", keep="first").set_index("_code")
        inc_by_code = inc_dedup["_inc"].to_dict()
        include_e = set(inc_dedup.index[inc_dedup["_inc"] < start_ts])
    else:
        include_e = None

    # a: 排除在 [成立+2年, 窗口 end_date] 内机构持仓连续两次 > 60% 的基金（不做 start_ts 裁剪）
    a_df = pd.read_csv(cyrjg_csv, dtype=str, encoding="utf-8-sig")
    date_col = "日期" if "日期" in a_df.columns else "公告日期"
    a_df = a_df.copy()
    a_df[date_col] = pd.to_datetime(a_df[date_col], errors="coerce")
    a_df = a_df[a_df[date_col] <= end_ts]
    a_df["_pct"] = a_df["机构持有比例"].map(_parse_pct)
    a_df["_code"] = a_df["基金代码"].map(_safe_code)
    a_df = a_df[a_df["_code"] != ""]

    exclude_a: set[str] = set()
    two_years = pd.DateOffset(years=2)
    for code, grp in a_df.groupby("_code"):
        inc = inc_by_code.get(code)
        if inc is None:
            continue
        cutoff = inc + two_years
        sub = grp[grp[date_col] >= cutoff]  # grp 已限制 date<=end_ts，无需重复判断
        if _has_consecutive_over_60(sub, date_col):
            exclude_a.add(code)
    codes -= exclude_a
    log.info("[eligible] a([成立+2年,end_date]内连续两次>60%%排除) 后 %d，排除 %d", len(codes), len(exclude_a))

    # b: 仅保留 end_date 前最新一条规模 > 2 亿的基金
    b_df = pd.read_csv(gmbd_csv, dtype=str, encoding="utf-8-sig")
    b_df["日期"] = pd.to_datetime(b_df["日期"], errors="coerce")
    b_df = b_df[b_df["日期"].notna() & (b_df["日期"] <= end_ts)]
    scale_col = "期末净资产（亿元）"
    if scale_col in b_df.columns:
        b_df["_scale"] = pd.to_numeric(b_df[scale_col], errors="coerce")
        b_df["_code"] = b_df["基金代码"].map(_safe_code)
        b_df = b_df[b_df["_code"] != ""].dropna(subset=["_scale"])
        # 同日期多行时 groupby.last 取最后一条，避免 idxmax 仅取首条
        latest = b_df.sort_values("日期").groupby("_code", as_index=False).last()
        include_b = set(latest[latest["_scale"] > 2]["_code"].tolist())  # _code 已由 _safe_code 规范化
        codes &= include_b
        log.info("[eligible] b(end_date前最新规模>2亿) 后 %d", len(codes))
    else:
        raise ValueError(f"missing column in gmbd csv: {scale_col}")

    # e: 仅保留 start_date 之前成立的基金（无法解析成立日期则跳过）
    if include_e is not None:
        codes &= include_e
        log.info("[eligible] e(start_date前成立) 后 %d", len(codes))
    else:
        log.warning("[eligible] e 无成立日期列，跳过该条件")

    base_codes = codes.copy()
    output_path = (output_path or (work_dir / "eligible_fund_candidates.csv")).resolve()
    cache_dir = output_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_path = cache_dir / f"eligible_base_{start_date}_{end_date}.csv"
    personnel_path = cache_dir / f"personnel_excluded_{start_date}_{end_date}.csv"

    # Base 缓存：不依赖 personnel_dir
    if base_path.exists():
        base_df = pd.read_csv(base_path, dtype=str, encoding="utf-8-sig")
        log.info("[eligible] base cache hit: %s", base_path)
    else:
        base_df = purchase_df[purchase_df["基金代码"].map(_safe_code).isin(base_codes)].copy()
        base_df.to_csv(base_path, index=False, encoding="utf-8-sig")
        log.info("[eligible] base cache write: %s", base_path)

    # f: 人事变动排除，拆分缓存，加载时合并
    personnel_excluded: set[str] = set()
    if personnel_dir is not None and personnel_dir.is_dir():
        if personnel_path.exists():
            excl_df = pd.read_csv(personnel_path, dtype=str, encoding="utf-8-sig")
            code_col = "基金编码" if "基金编码" in excl_df.columns else "基金代码"
            personnel_excluded = {_safe_code(c) for c in excl_df[code_col].dropna().tolist()}
            log.info("[eligible] f personnel cache hit: %s，排除 %d", personnel_path, len(personnel_excluded))
        else:
            personnel_start = end_ts - pd.DateOffset(years=1)
            personnel_excluded = _compute_personnel_excluded(
                set(base_df["基金代码"].map(_safe_code)), personnel_dir, personnel_start, end_ts
            )
            pd.DataFrame({"基金编码": sorted(personnel_excluded)}).to_csv(
                personnel_path, index=False, encoding="utf-8-sig"
            )
            log.info("[eligible] f([end-1年,end]内人事变动排除) cache write: 排除 %d", len(personnel_excluded))
    else:
        if personnel_dir is not None:
            log.warning("[eligible] f 人事目录不存在或非目录，跳过: %s", personnel_dir)

    final_codes = set(base_df["基金代码"].map(_safe_code)) - personnel_excluded
    result = base_df[base_df["基金代码"].map(_safe_code).isin(final_codes)].copy()
    log.info("[eligible] 最终结果 %d 只 (base %d - personnel_excluded %d)", len(result), len(base_codes), len(personnel_excluded))

    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info("[eligible] 写入 %s", output_path)
    return output_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="v2 预备 eligible 按窗口计算（仅使用本地缓存）")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--personnel-dir", type=Path, default=None, help="人事变动目录 fund_personnel_by_code，传入后排除[end-1年,end]内有人事变动的基金")
    args = parser.parse_args()

    try:
        run(
            work_dir=args.work_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            personnel_dir=args.personnel_dir,
            output_path=args.output,
            logger=logger,
        )
        return 0
    except Exception as e:  # noqa: BLE001
        logger.exception("%s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
