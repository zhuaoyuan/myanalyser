#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预备数据工作流：从指定日期起，拉取并筛选基金预备数据。

流程：
1. 获取全体基金购买 (x)
2. 根据 x 获取持有人比例全历史 (a)
3. 根据 x 获取基金规模全历史 (b)
4. 根据 x 获取基金费率全历史 (c)
5. 根据 c 进行基金分类 (c.1)
6. 根据 x 获取全体基金详情 (e)
7. 筛选：x + c.1(存在) + a(date后且成立>2年后机构持仓连续两次>60%则排除) + b(曾规模>2亿) + e(date前成立) -> d.1

样例：
python myanalyser/tools/prep/prep_data_workflow.py \
  --date 2021-01-01 \
  -o myanalyser/tmp/1/prep_result_m.csv \
  --purchase-csv finance-runs/run_20260310_191534/data/versions/20260310_191534/fund_etl/fund_purchase.csv \
  --cyrjg-csv finance-runs/run_20260310_191534/data/versions/20260310_191534/fund_etl/cyrjg_out.csv \
  --gmbd-csv /Users/zhuaoyuan/cursor-workspace/finance/myanalyser/tmp/prep_work/fund_gmbd.csv \
  --fee-csv finance-runs/run_20260310_191534/data/versions/20260310_191534/fund_etl/fee/fund_fee_complete_fixed.csv \
  --overview-csv finance-runs/run_20260301_1_formal_retry_step4_rerun/data/versions/20260301_1_formal_retry_step4_rerun/fund_etl/fund_overview.csv

"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# 项目路径：tools/prep 的 parent.parent 为 myanalyser
_PREP_DIR = Path(__file__).resolve().parent
_MYANALYSER = _PREP_DIR.parent.parent
assert _MYANALYSER.name == "myanalyser", f"unexpected _MYANALYSER: {_MYANALYSER}"
if str(_MYANALYSER / "src") not in sys.path:
    sys.path.insert(0, str(_MYANALYSER / "src"))


def _safe_code(v: object) -> str:
    return str(v).strip().zfill(6)


def _parse_date(text: object) -> pd.Timestamp | None:
    """解析日期，支持 2013年03月20日、YYYY-MM-DD 等格式。"""
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


def _has_consecutive_over_60(grp: pd.DataFrame, date_col: str) -> bool:
    """该基金是否存在连续两次机构持仓>60%。"""
    df = grp.dropna(subset=["_pct", date_col]).sort_values(date_col)
    if len(df) < 2:
        return False
    vals = df["_pct"].values
    for i in range(len(vals) - 1):
        if vals[i] > 60 and vals[i + 1] > 60:
            return True
    return False


def _parse_pct(val: object) -> float | None:
    """解析百分比，如 63.45% -> 63.45。"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s == "---":
        return None
    m = re.match(r"([\d.]+)\s*%", s)
    return float(m[1]) if m else None


def _run_cli(script: str, args: list[str], logger: logging.Logger, stream_output: bool = True) -> bool:
    """执行 CLI 脚本，返回是否成功。

    cwd 设为 myanalyser 父目录（工作区根），子脚本路径使用绝对路径，保证任意工作目录下调用均可靠。
    stream_output=True 时子进程 stdout/stderr 直接输出到终端，便于查看实时进度。
    """
    script_path = (_PREP_DIR / script).resolve()
    cmd = [sys.executable, str(script_path)] + [str(a) for a in args]
    logger.info("执行: %s", " ".join(cmd))
    ret = subprocess.run(cmd, capture_output=not stream_output, text=True, cwd=_MYANALYSER.parent)
    if ret.returncode != 0:
        err_msg = (ret.stderr or ret.stdout or "") if not stream_output else "（请查看上方子进程输出）"
        logger.error("退出码 %d: %s", ret.returncode, err_msg)
        return False
    return True


def _step1_purchase(work_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """获取全体基金购买 x。"""
    from fund_etl import run_step1_purchase

    logger.info("[x] 开始获取基金购买列表")
    out = work_dir / "fund_purchase.csv"
    df = run_step1_purchase(out)
    logger.info("[x] 完成 基金购买 %d 行 -> %s", len(df), out)
    return df


def _step_a_cyrjg(
    purchase_csv: Path,
    work_dir: Path,
    existing_csv: Path | None,
    delay: float,
    logger: logging.Logger,
) -> Path:
    """获取持有人比例全历史 a。若 existing_csv 传入则直接使用。"""
    if existing_csv and existing_csv.exists():
        logger.info("[a] 使用已有持有人比例: %s", existing_csv)
        return existing_csv

    n_codes = len(pd.read_csv(purchase_csv, dtype={"基金代码": str})["基金代码"].dropna().unique())
    logger.info("[a] 开始抓取持有人比例，共 %d 只基金", n_codes)
    out = work_dir / "fund_cyrjg.csv"
    ok = _run_cli(
        "fetch_fund_cyrjg.py",
        ["-i", str(purchase_csv), "-o", str(out), "--delay", str(delay)],
        logger,
    )
    if not ok or not out.exists():
        raise FileNotFoundError("持有人比例抓取失败")
    logger.info("[a] 完成 持有人比例 -> %s", out)
    return out


def _step_b_gmbd(
    purchase_csv: Path,
    work_dir: Path,
    existing_csv: Path | None,
    delay: float,
    logger: logging.Logger,
) -> Path:
    """获取基金规模全历史 b。若 existing 传入则合并增量。"""
    codes_df = pd.read_csv(purchase_csv, dtype={"基金代码": str})
    codes = set(codes_df["基金代码"].dropna().map(_safe_code).tolist())

    if existing_csv and existing_csv.exists():
        existing = pd.read_csv(existing_csv, dtype=str, encoding="utf-8-sig")
        if "基金代码" in existing.columns:
            done_codes = set(existing["基金代码"].dropna().map(_safe_code).tolist())
            to_fetch = sorted(codes - done_codes)
        else:
            to_fetch = sorted(codes)
    else:
        existing = None
        to_fetch = sorted(codes)

    out = work_dir / "fund_gmbd.csv"
    if to_fetch:
        if existing is not None and "基金代码" in existing.columns:
            done_codes = set(existing["基金代码"].dropna().map(_safe_code).tolist())
            logger.info("[b] 开始抓取基金规模，待抓取 %d 只，已有 %d 只", len(to_fetch), len(done_codes))
        else:
            logger.info("[b] 开始抓取基金规模，共 %d 只", len(to_fetch))
        tmp_purchase = work_dir / "_tmp_gmbd_purchase.csv"
        tmp_df = pd.DataFrame({"基金代码": to_fetch})
        tmp_df.to_csv(tmp_purchase, index=False, encoding="utf-8-sig")
        ok = _run_cli(
            "fetch_fund_gmbd.py",
            ["-i", str(tmp_purchase), "-o", str(out), "--delay", str(delay)],
            logger,
        )
        if not ok:
            raise RuntimeError("基金规模抓取失败")
        fetched = pd.read_csv(out, dtype=str, encoding="utf-8-sig") if out.exists() else pd.DataFrame()
        if existing is not None and not fetched.empty:
            merged = pd.concat([existing, fetched], ignore_index=True)
            if "基金代码" in merged.columns and "日期" in merged.columns:
                merged = merged.drop_duplicates(subset=["基金代码", "日期"], keep="first")
            merged.to_csv(out, index=False, encoding="utf-8-sig")
        elif existing is not None:
            existing.to_csv(out, index=False, encoding="utf-8-sig")
    elif existing is not None:
        existing.to_csv(out, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(out, index=False, encoding="utf-8-sig")

    logger.info("[b] 完成 基金规模 -> %s", out)
    return out


def _step_c_fee(
    purchase_csv: Path,
    work_dir: Path,
    existing_csv: Path | None,
    delay: float,
    logger: logging.Logger,
) -> Path:
    """获取基金费率全历史 c。若 existing 传入则合并增量。"""
    codes_df = pd.read_csv(purchase_csv, dtype=str)
    codes = set(codes_df["基金代码"].dropna().map(_safe_code).tolist())

    if existing_csv and existing_csv.exists():
        existing = pd.read_csv(existing_csv, dtype=str, encoding="utf-8-sig")
        code_col = "基金编码" if "基金编码" in existing.columns else "基金代码"
        done_codes = set(existing[code_col].dropna().map(_safe_code).tolist())
        to_fetch = sorted(codes - done_codes)
    else:
        existing = None
        to_fetch = sorted(codes) if codes else []

    out = work_dir / "fund_fee_structured.csv"
    if not to_fetch:
        if existing is not None:
            logger.info("[c] 费率已完整，复用 %s", existing_csv)
            existing.to_csv(out, index=False, encoding="utf-8-sig")
            return out
        pd.DataFrame().to_csv(out, index=False, encoding="utf-8-sig")
        logger.info("[c] 基金列表为空，跳过费率抓取")
        return out

    purchase_for_fee = purchase_csv
    if to_fetch:
        logger.info("[c] 开始抓取基金费率，待抓取 %d 只（已有 %d 只）", len(to_fetch), len(codes) - len(to_fetch) if existing is not None else 0)
        tmp_purchase = work_dir / "_tmp_fee_purchase.csv"
        sub = codes_df[codes_df["基金代码"].map(_safe_code).isin(to_fetch)]
        sub.to_csv(tmp_purchase, index=False, encoding="utf-8-sig")
        purchase_for_fee = tmp_purchase

    ok = _run_cli(
        "fetch_fund_fee.py",
        [str(purchase_for_fee), "-o", str(out), "--delay", str(delay)],
        logger,
    )
    if not ok:
        raise RuntimeError("基金费率抓取失败")

    if existing is not None and out.exists():
        fetched = pd.read_csv(out, dtype=str, encoding="utf-8-sig")
        code_col = "基金编码" if "基金编码" in existing.columns else "基金代码"
        merged = pd.concat([existing, fetched], ignore_index=True)
        merged.to_csv(out, index=False, encoding="utf-8-sig")

    logger.info("[c] 完成 基金费率 -> %s", out)
    return out


def _step_c1_filter_fee(fee_csv: Path, work_dir: Path, logger: logging.Logger) -> Path:
    """根据费率进行基金分类 c.1。"""
    n_rows = len(pd.read_csv(fee_csv, dtype=str)) if fee_csv.exists() else 0
    logger.info("[c.1] 开始基金费率分类，输入 %d 行", n_rows)
    out = work_dir / "fund_fee_filtered.csv"
    ok = _run_cli("filter_fund_fee_by_holding.py", [str(fee_csv), "-o", str(out)], logger)
    if not ok:
        raise RuntimeError("基金费率分类失败")
    logger.info("[c.1] 完成 基金分类 -> %s", out)
    return out


def _step_e_overview(
    purchase_csv: Path,
    work_dir: Path,
    existing_csv: Path | None,
    logger: logging.Logger,
) -> Path:
    """获取全体基金详情 e。run_step2_overview 本身支持增量。"""
    import shutil

    from fund_etl import RetryConfig, run_step2_overview

    out = work_dir / "fund_overview.csv"
    fail_log = work_dir / "failed_overview.jsonl"
    retry_cfg = RetryConfig()
    if existing_csv and existing_csv.exists():
        shutil.copy2(existing_csv, out)
        logger.info("[e] 从已有文件初始化: %s", existing_csv)
    n_codes = len(pd.read_csv(purchase_csv, dtype=str)["基金代码"].dropna().unique())
    logger.info("[e] 开始获取基金详情，共 %d 只基金（run_step2_overview 将按增量打印进度）", n_codes)
    summary = run_step2_overview(
        purchase_csv=purchase_csv,
        overview_csv=out,
        fail_log=fail_log,
        retry_cfg=retry_cfg,
    )
    logger.info("[e] 完成 基金详情 %s -> %s", summary, out)
    return out


def _apply_filters(
    purchase_df: pd.DataFrame,
    date_str: str,
    cyrjg_csv: Path,
    gmbd_csv: Path,
    fee_filtered_csv: Path,
    overview_csv: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """应用筛选条件，得到 d.1。"""
    logger.info("[d.1] 开始应用筛选条件，起始日期 %s，候选 %d 只", date_str, len(set(purchase_df["基金代码"].dropna().map(_safe_code).tolist())))
    date_ts = pd.to_datetime(date_str)
    codes = set(purchase_df["基金代码"].dropna().map(_safe_code).tolist())

    # 统一读取 overview，供 a、e 共用，避免重复 IO
    inc_by_code: dict[str, pd.Timestamp] = {}
    include_e: set[str] | None = None
    if overview_csv.exists():
        e_df = pd.read_csv(overview_csv, dtype=str)
        col = "成立日期/规模" if "成立日期/规模" in e_df.columns else "成立日期"
        if col in e_df.columns:
            e_df = e_df.copy()
            e_df["_code"] = e_df["基金代码"].map(_safe_code)
            e_df["_inc"] = e_df[col].map(_parse_date)
            inc_ok = e_df[e_df["_inc"].notna() & (e_df["_code"] != "")]
            inc_ok = inc_ok.sort_values("_inc", na_position="last")
            # 与 inc_by_code 一致：每基金取最早成立日，再筛 date 前成立
            inc_dedup = inc_ok.drop_duplicates("_code", keep="first").set_index("_code")
            inc_by_code = inc_dedup["_inc"].to_dict()
            include_e = set(inc_dedup.index[inc_dedup["_inc"] < date_ts])
        else:
            include_e = None  # 无成立日期列时跳过 e 条件

    # c.1: 必须在 c.1 中存在
    if fee_filtered_csv.exists():
        c1_df = pd.read_csv(fee_filtered_csv, dtype=str)
        code_col = "基金编码" if "基金编码" in c1_df.columns else "基金代码"
        c1_codes = set(c1_df[code_col].dropna().map(_safe_code).tolist())
        codes &= c1_codes
        logger.info("[筛选] c.1 后 %d", len(codes))
    else:
        logger.warning("c.1 文件不存在，跳过该条件")

    # a: 排除 date 之后、且基金成立大于2年后、机构持仓比例连续两次超过60%的基金
    if cyrjg_csv.exists():
        a_df = pd.read_csv(cyrjg_csv, dtype=str)
        date_col = "日期" if "日期" in a_df.columns else "公告日期"
        a_df = a_df.copy()
        a_df[date_col] = pd.to_datetime(a_df[date_col], errors="coerce")
        a_df = a_df[a_df[date_col] >= date_ts]  # date 之后
        a_df["_pct"] = a_df["机构持有比例"].map(_parse_pct)
        a_df["_code"] = a_df["基金代码"].map(_safe_code)
        a_df = a_df[a_df["_code"] != ""]  # 过滤空 code 避免无效迭代

        exclude_a: set[str] = set()
        two_years = pd.DateOffset(years=2)
        for code, grp in a_df.groupby("_code"):
            inc = inc_by_code.get(code)
            if inc is None:
                continue  # 无成立日期则无法判断「成立>2年后」，不排除
            cutoff = inc + two_years
            sub = grp[grp[date_col] >= cutoff]
            if _has_consecutive_over_60(sub, date_col):
                exclude_a.add(code)
        codes -= exclude_a
        logger.info("[筛选] a(date后+成立>2年后+连续两次>60%%排除) 后 %d，排除 %d", len(codes), len(exclude_a))
    else:
        logger.warning("a 文件不存在，跳过该条件")

    # b: 仅保留 date 至今发生过规模 > 2亿 的基金
    if gmbd_csv.exists():
        b_df = pd.read_csv(gmbd_csv, dtype=str)
        b_df["日期"] = pd.to_datetime(b_df["日期"], errors="coerce")
        b_df = b_df[b_df["日期"] >= date_ts]
        scale_col = "期末净资产（亿元）"
        if scale_col in b_df.columns:
            b_df["_scale"] = pd.to_numeric(b_df[scale_col], errors="coerce")
            include_b = set(b_df[b_df["_scale"] > 2]["基金代码"].dropna().map(_safe_code).tolist())
            codes &= include_b
            logger.info("[筛选] b(规模>2亿) 后 %d", len(codes))
    else:
        logger.warning("b 文件不存在，跳过该条件")

    # e: 仅保留 date 之前成立的基金（无法解析成立日期的排除），复用开头读取的 include_e
    if include_e is not None:
        codes &= include_e
        logger.info("[筛选] e(date前成立) 后 %d", len(codes))
    elif not overview_csv.exists():
        logger.warning("e 文件不存在，跳过该条件")
    else:
        logger.warning("e 无成立日期列，跳过该条件")

    result = purchase_df[purchase_df["基金代码"].map(_safe_code).isin(codes)].copy()
    logger.info("[d.1] 完成筛选，剩余 %d 只", len(result))
    return result


def run(
    date_str: str,
    output_path: Path,
    work_dir: Path,
    *,
    purchase_csv: Path | None = None,
    cyrjg_csv: Path | None = None,
    gmbd_csv: Path | None = None,
    fee_csv: Path | None = None,
    overview_csv: Path | None = None,
    delay: float = 0.3,
    logger: logging.Logger | None = None,
) -> Path:
    """执行预备数据工作流。"""
    log = logger or logging.getLogger(__name__)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. x
    if purchase_csv and purchase_csv.exists():
        x_df = pd.read_csv(purchase_csv, dtype=str)
        x_path = work_dir / "fund_purchase.csv"
        x_df.to_csv(x_path, index=False, encoding="utf-8-sig")
        log.info("[x] 使用已有购买: %s", purchase_csv)
    else:
        x_df = _step1_purchase(work_dir, log)
        x_path = work_dir / "fund_purchase.csv"

    # 2–4. a, b, c
    a_path = _step_a_cyrjg(x_path, work_dir, cyrjg_csv, delay, log)
    b_path = _step_b_gmbd(x_path, work_dir, gmbd_csv, delay, log)
    c_path = _step_c_fee(x_path, work_dir, fee_csv, delay, log)

    # 5. c.1
    c1_path = _step_c1_filter_fee(c_path, work_dir, log)

    # 6. e
    e_path = _step_e_overview(x_path, work_dir, overview_csv, log)

    # 7. 筛选 -> d.1
    d1_df = _apply_filters(x_df, date_str, a_path, b_path, c1_path, e_path, log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    d1_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info("[d.1] 最终结果 %d 行 -> %s", len(d1_df), output_path)
    return output_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="预备数据工作流：从指定日期起拉取并筛选基金预备数据")
    parser.add_argument("--date", required=True, help="筛选起始日期，格式 YYYY-MM-DD")
    parser.add_argument("-o", "--output", type=Path, required=True, help="最终结果文件路径 (d.1)")
    parser.add_argument("--work-dir", type=Path, default=None, help="工作目录，默认与 output 同目录下的 prep_work")
    parser.add_argument("--purchase-csv", type=Path, default=None, help="已有基金购买 CSV，传入则跳过 step1")
    parser.add_argument("--cyrjg-csv", type=Path, default=None, help="已有持有人比例 CSV，传入则直接使用")
    parser.add_argument("--gmbd-csv", type=Path, default=None, help="已有规模 CSV，传入则增量查询")
    parser.add_argument("--fee-csv", type=Path, default=None, help="已有费率 CSV，传入则增量查询")
    parser.add_argument("--overview-csv", type=Path, default=None, help="已有基金详情 CSV，传入则增量查询")
    parser.add_argument("--delay", type=float, default=0.1, help="请求间隔秒数")
    args = parser.parse_args()

    output_path = args.output.resolve()
    work_dir = (args.work_dir or output_path.parent / "prep_work").resolve()

    try:
        run(
            date_str=args.date,
            output_path=output_path,
            work_dir=work_dir,
            purchase_csv=args.purchase_csv.resolve() if args.purchase_csv else None,
            cyrjg_csv=args.cyrjg_csv.resolve() if args.cyrjg_csv else None,
            gmbd_csv=args.gmbd_csv.resolve() if args.gmbd_csv else None,
            fee_csv=args.fee_csv.resolve() if args.fee_csv else None,
            overview_csv=args.overview_csv.resolve() if args.overview_csv else None,
            delay=args.delay,
            logger=logger,
        )
        return 0
    except Exception as e:
        logger.exception("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
