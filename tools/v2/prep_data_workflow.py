#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 预备数据工作流（宽存）：只抓取全量原始数据，不做筛选。

命令样例：
python myanalyser/tools/v2/prep_data_workflow.py \
  --work-dir myanalyser/tmp/prep_work_v2 \
  --purchase-csv myanalyser/tmp/1_refilter/prep_work/fund_purchase.csv \
  --cyrjg-csv finance-runs/run_20260310_191534/data/versions/20260310_191534/fund_etl/cyrjg_out.csv \
  --gmbd-csv myanalyser/tmp/1_refilter/prep_work/fund_gmbd.csv \
  --fee-csv myanalyser/tmp/1_refilter/prep_work/fund_fee_structured.csv \
  --overview-csv myanalyser/tmp/1_refilter/prep_work/fund_overview.csv \
  --delay 0

产出到 work_dir（固定文件名）：
- fund_purchase.csv
- fund_cyrjg.csv
- fund_gmbd.csv
- fund_fee_structured.csv
- fund_overview.csv
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

_TOOLS_V2_DIR = Path(__file__).resolve().parent
_MYANALYSER = _TOOLS_V2_DIR.parent.parent
assert _MYANALYSER.name == "myanalyser", f"unexpected _MYANALYSER: {_MYANALYSER}"
_PREP_DIR = _MYANALYSER / "tools" / "prep"

if str(_MYANALYSER / "src") not in sys.path:
    sys.path.insert(0, str(_MYANALYSER / "src"))


def _safe_code(v: object) -> str:
    return str(v).strip().zfill(6)


def _run_cli(script: str, args: list[str], logger: logging.Logger, stream_output: bool = True) -> bool:
    """执行 tools/prep 下的 CLI 脚本。"""
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
    out = work_dir / "fund_cyrjg.csv"
    if existing_csv and existing_csv.exists():
        logger.info("[a] 使用已有持有人比例: %s", existing_csv)
        shutil.copy2(existing_csv, out)
        return out

    n_codes = len(pd.read_csv(purchase_csv, dtype={"基金代码": str})["基金代码"].dropna().unique())
    logger.info("[a] 开始抓取持有人比例，共 %d 只基金", n_codes)
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
    tmp_out = work_dir / "_tmp_gmbd_fetched.csv"
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
            ["-i", str(tmp_purchase), "-o", str(tmp_out), "--delay", str(delay)],
            logger,
        )
        if not ok:
            raise RuntimeError("基金规模抓取失败")
        if not tmp_out.exists():
            raise FileNotFoundError(f"fund_gmbd 输出缺失: {tmp_out}")
        fetched = pd.read_csv(tmp_out, dtype=str, encoding="utf-8-sig")
        if existing is not None and not fetched.empty:
            merged = pd.concat([existing, fetched], ignore_index=True)
            if "基金代码" in merged.columns and "日期" in merged.columns:
                merged = merged.drop_duplicates(subset=["基金代码", "日期"], keep="first")
            merged.to_csv(out, index=False, encoding="utf-8-sig")
        elif existing is not None:
            existing.to_csv(out, index=False, encoding="utf-8-sig")
        else:
            fetched.to_csv(out, index=False, encoding="utf-8-sig")
        if tmp_out.exists():
            tmp_out.unlink()
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
    tmp_out = work_dir / "_tmp_fee_fetched.csv"
    if not to_fetch:
        if existing is not None:
            existing.to_csv(out, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame().to_csv(out, index=False, encoding="utf-8-sig")
        logger.info("[c] 无需抓取费率，已写空/复用文件 -> %s", out)
        return out

    tmp_purchase = work_dir / "_tmp_fee_purchase.csv"
    tmp_df = pd.DataFrame({"基金代码": to_fetch})
    tmp_df.to_csv(tmp_purchase, index=False, encoding="utf-8-sig")
    ok = _run_cli(
        "fetch_fund_fee.py",
        ["-i", str(tmp_purchase), "-o", str(tmp_out), "--delay", str(delay)],
        logger,
    )
    if not ok:
        raise RuntimeError("基金费率抓取失败")
    if not tmp_out.exists():
        raise FileNotFoundError(f"fund_fee 输出缺失: {tmp_out}")

    fetched = pd.read_csv(tmp_out, dtype=str, encoding="utf-8-sig") if tmp_out.exists() else pd.DataFrame()
    if existing is not None and not fetched.empty:
        merged = pd.concat([existing, fetched], ignore_index=True)
        merged.to_csv(out, index=False, encoding="utf-8-sig")
    elif existing is not None:
        existing.to_csv(out, index=False, encoding="utf-8-sig")
    else:
        fetched.to_csv(out, index=False, encoding="utf-8-sig")
    if tmp_out.exists():
        tmp_out.unlink()

    logger.info("[c] 完成 基金费率 -> %s", out)
    return out


def _step_e_overview(
    purchase_csv: Path,
    work_dir: Path,
    existing_csv: Path | None,
    logger: logging.Logger,
) -> Path:
    """获取全体基金详情 e。run_step2_overview 本身支持增量。"""
    from fund_etl import RetryConfig, run_step2_overview

    out = work_dir / "fund_overview.csv"
    fail_log = work_dir / "failed_overview.jsonl"
    retry_cfg = RetryConfig()
    if existing_csv and existing_csv.exists():
        shutil.copy2(existing_csv, out)
        # run_step2_overview 以 overview_csv 为增量基底
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


def run(
    work_dir: Path,
    *,
    purchase_csv: Path | None = None,
    cyrjg_csv: Path | None = None,
    gmbd_csv: Path | None = None,
    fee_csv: Path | None = None,
    overview_csv: Path | None = None,
    delay: float = 0.3,
    logger: logging.Logger | None = None,
) -> None:
    """执行 v2 预备数据宽存流程（仅 L1）。"""
    log = logger or logging.getLogger(__name__)
    work_dir.mkdir(parents=True, exist_ok=True)

    if purchase_csv and purchase_csv.exists():
        x_df = pd.read_csv(purchase_csv, dtype=str)
        x_path = work_dir / "fund_purchase.csv"
        x_df.to_csv(x_path, index=False, encoding="utf-8-sig")
        log.info("[x] 使用已有购买: %s", purchase_csv)
    else:
        _step1_purchase(work_dir, log)
        x_path = work_dir / "fund_purchase.csv"

    _step_a_cyrjg(x_path, work_dir, cyrjg_csv, delay, log)
    _step_b_gmbd(x_path, work_dir, gmbd_csv, delay, log)
    _step_c_fee(x_path, work_dir, fee_csv, delay, log)
    _step_e_overview(x_path, work_dir, overview_csv, log)
    log.info("[v2] L1 数据准备完成 -> %s", work_dir)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="v2 预备数据工作流（宽存）：只抓取全量原始数据")
    parser.add_argument("--work-dir", type=Path, default=None, help="工作目录，默认 myanalyser/tmp/prep_work_v2")
    parser.add_argument("--purchase-csv", type=Path, default=None, help="已有基金购买 CSV（必须存在），传入则跳过 step1")
    parser.add_argument("--cyrjg-csv", type=Path, default=None, help="已有持有人比例 CSV（必须存在），传入则直接使用")
    parser.add_argument("--gmbd-csv", type=Path, default=None, help="已有规模 CSV（必须存在），传入则增量查询")
    parser.add_argument("--fee-csv", type=Path, default=None, help="已有费率 CSV（必须存在），传入则增量查询")
    parser.add_argument("--overview-csv", type=Path, default=None, help="已有基金详情 CSV（必须存在），传入则增量查询")
    parser.add_argument("--delay", type=float, default=0.1, help="请求间隔秒数")
    args = parser.parse_args()

    work_dir = (args.work_dir or (_MYANALYSER / "tmp" / "prep_work_v2")).resolve()

    def _resolve_existing(path: Path | None, label: str) -> Path | None:
        if path is None:
            return None
        path = path.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        return path.resolve()

    try:
        run(
            work_dir=work_dir,
            purchase_csv=_resolve_existing(args.purchase_csv, "purchase_csv"),
            cyrjg_csv=_resolve_existing(args.cyrjg_csv, "cyrjg_csv"),
            gmbd_csv=_resolve_existing(args.gmbd_csv, "gmbd_csv"),
            fee_csv=_resolve_existing(args.fee_csv, "fee_csv"),
            overview_csv=_resolve_existing(args.overview_csv, "overview_csv"),
            delay=args.delay,
            logger=logger,
        )
        return 0
    except Exception as e:
        logger.exception("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
