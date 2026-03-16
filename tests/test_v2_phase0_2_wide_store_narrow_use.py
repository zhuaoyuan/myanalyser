# -*- coding: utf-8 -*-
"""
v2 宽存窄用 Phase0-2 改造 单元测试。

需求来源：myanalyser/docs/需求日志/20260315_宽存窄用_v2_phase0-2改造.md
补充需求：myanalyser/docs/需求日志/20260316_基金过滤逻辑与预期差异调整.md

场景分类：
- 正常场景：窗口化 Compare、Filter v2（支持 end-date）、prep_eligible_window
- 异常场景：缺失目录、start>end、必填参数缺失
- 边界条件：空值、空文件、极大值、无公共日期、偏差阈值临界、规模/机构持仓临界值
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest


# ========== 场景清单（供文档与报告引用） ==========
# 需求来源: 20260316_基金过滤逻辑与预期差异调整
SCENARIOS = """
## 正常场景
- compare_window: 窗口内有公共日期，产出 summary + details + error_jsonl
- compare_window_single_fund: 单基金完整比对，偏差分桶正确
- filter_v2_all_pass: 规则1-5全通过，基金不被过滤
- filter_v2_rules_1_to_5: 规则1/2/3/4/5 分别触发过滤
- filter_v2_end_date: 使用 [start_date, end_date] 判定规则4/5
- filter_extended_end_date: 调用方传入延伸 end_date，compare/integrity 使用延伸区间
- prep_eligible_window: c1+b+e 通过，产出 eligible csv
- prep_rule_a: [成立+2年, end_date] 内连续两次>60% 排除（含 start_ts 前数据）
- prep_rule_b: 仅用 end_date 前最新规模>2亿 判定

## 异常场景
- compare_missing_adjusted_dir: base_dir 缺少 fund_adjusted_nav_by_code → FileNotFoundError
- compare_missing_cum_dir: base_dir 缺少 fund_cum_return_by_code → FileNotFoundError
- compare_start_after_end: start_date > end_date → ValueError
- filter_start_after_end: start_date > end_date → ValueError
- prep_eligible_missing_purchase: 缺少 fund_purchase.csv → FileNotFoundError
- prep_eligible_missing_cyrjg: 缺少 fund_cyrjg.csv → FileNotFoundError
- prep_eligible_missing_gmbd: 缺少 fund_gmbd.csv → FileNotFoundError
- prep_eligible_start_after_end: start_date > end_date → ValueError

## 边界条件
- compare_empty_dirs: 两目录无 csv，all_codes 空，产出空 summary
- compare_no_common_date: 窗口内无公共日期，记录 数据是否缺失=是
- compare_parse_error: CSV 缺列或格式错误，记录到 error jsonl
- filter_empty_purchase: 申购列表为空
- filter_compare_detail_outside_window: 规则4 区间外记录不参与判定
- filter_deviation_at_threshold: 偏差绝对值等于 max_abs_deviation 视为过滤（>=）
- filter_integrity_empty_dir: integrity_details_dir 为空目录
- filter_extended_end_date_detail_in_range: 延伸区间内 compare 记录参与判定
- prep_eligible_empty_result: 所有基金被过滤，输出空 csv
- prep_rule_a_exactly_60: 恰好60% 不算>60%，不排除
- prep_rule_a_consecutive_before_start: [成立+2年, start_ts) 内连续两次>60% 也排除
- prep_rule_b_scale_exactly_2: 最新规模恰好2亿 不保留（>2亿才保留）
- prep_rule_b_no_scale_before_end: end_date 前无规模数据 排除
- prep_rule_b_same_date_last: 同日期多行取 last
"""


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ========== compare_adjusted_nav_and_cum_return_window ==========


class TestCompareAdjustedNavAndCumReturnWindow:
    """窗口化 Compare 模块测试"""

    def test_normal_window_with_common_dates(self, tmp_path: Path) -> None:
        """正常：窗口内有公共日期，产出 summary + details"""
        base = tmp_path / "base"
        adjusted = base / "fund_adjusted_nav_by_code"
        cum = base / "fund_cum_return_by_code"
        adjusted.mkdir(parents=True)
        cum.mkdir(parents=True)

        _write_csv(
            adjusted / "000001.csv",
            pd.DataFrame({
                "净值日期": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "复权净值": [1.0, 1.02, 1.05],
            }),
        )
        _write_csv(
            cum / "000001.csv",
            pd.DataFrame({
                "日期": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "累计收益率": [0.0, 2.0, 5.0],
            }),
        )
        out_dir = tmp_path / "out"
        from v2.compare.compare_adjusted_nav_and_cum_return_window import (
            compare_adjusted_nav_and_cum_return_window,
        )
        result = compare_adjusted_nav_and_cum_return_window(
            base_dir=base,
            start_date="2024-01-01",
            end_date="2024-01-03",
            output_dir=out_dir,
        )
        assert result["summary_csv"].exists()
        summary = pd.read_csv(result["summary_csv"], dtype={"基金代码": str})
        assert len(summary) == 1
        assert summary.iloc[0]["基金代码"] == "000001"
        assert summary.iloc[0]["数据是否缺失"] == "否"
        assert int(summary.iloc[0]["参与比对收益率的天数"]) >= 1
        assert (result["detail_dir"] / "000001.csv").exists()

    def test_missing_adjusted_dir_raises(self, tmp_path: Path) -> None:
        """异常：缺少 fund_adjusted_nav_by_code 目录"""
        base = tmp_path / "base"
        cum = base / "fund_cum_return_by_code"
        cum.mkdir(parents=True)
        from v2.compare.compare_adjusted_nav_and_cum_return_window import (
            compare_adjusted_nav_and_cum_return_window,
        )
        with pytest.raises(FileNotFoundError, match="fund_adjusted_nav_by_code"):
            compare_adjusted_nav_and_cum_return_window(
                base_dir=base,
                start_date="2024-01-01",
                end_date="2024-01-31",
            )

    def test_missing_cum_return_dir_raises(self, tmp_path: Path) -> None:
        """异常：缺少 fund_cum_return_by_code 目录"""
        base = tmp_path / "base"
        adjusted = base / "fund_adjusted_nav_by_code"
        adjusted.mkdir(parents=True)
        from v2.compare.compare_adjusted_nav_and_cum_return_window import (
            compare_adjusted_nav_and_cum_return_window,
        )
        with pytest.raises(FileNotFoundError, match="fund_cum_return_by_code"):
            compare_adjusted_nav_and_cum_return_window(
                base_dir=base,
                start_date="2024-01-01",
                end_date="2024-01-31",
            )

    def test_start_after_end_raises(self, tmp_path: Path) -> None:
        """异常：start_date > end_date"""
        base = tmp_path / "base"
        (base / "fund_adjusted_nav_by_code").mkdir(parents=True)
        (base / "fund_cum_return_by_code").mkdir(parents=True)
        _write_csv(base / "fund_adjusted_nav_by_code" / "000001.csv", pd.DataFrame({"净值日期": ["2024-01-01"], "复权净值": [1.0]}))
        _write_csv(base / "fund_cum_return_by_code" / "000001.csv", pd.DataFrame({"日期": ["2024-01-01"], "累计收益率": [0.0]}))
        from v2.compare.compare_adjusted_nav_and_cum_return_window import (
            compare_adjusted_nav_and_cum_return_window,
        )
        with pytest.raises(ValueError, match="start-date cannot be after end-date"):
            compare_adjusted_nav_and_cum_return_window(
                base_dir=base,
                start_date="2024-02-01",
                end_date="2024-01-01",
            )

    def test_no_common_date_in_window_logged(self, tmp_path: Path) -> None:
        """边界：窗口内无公共日期，记录 数据是否缺失=是"""
        base = tmp_path / "base"
        adjusted = base / "fund_adjusted_nav_by_code"
        cum = base / "fund_cum_return_by_code"
        adjusted.mkdir(parents=True)
        cum.mkdir(parents=True)
        _write_csv(
            adjusted / "000001.csv",
            pd.DataFrame({"净值日期": ["2024-01-01", "2024-01-02"], "复权净值": [1.0, 1.02]}),
        )
        _write_csv(
            cum / "000001.csv",
            pd.DataFrame({"日期": ["2024-01-10", "2024-01-11"], "累计收益率": [0.1, 0.12]}),
        )
        out_dir = tmp_path / "out"
        from v2.compare.compare_adjusted_nav_and_cum_return_window import (
            compare_adjusted_nav_and_cum_return_window,
        )
        result = compare_adjusted_nav_and_cum_return_window(
            base_dir=base,
            start_date="2024-01-01",
            end_date="2024-01-15",
            output_dir=out_dir,
        )
        summary = pd.read_csv(result["summary_csv"], dtype={"基金代码": str})
        assert summary.iloc[0]["数据是否缺失"] == "是"
        if result["error_jsonl"].exists():
            lines = result["error_jsonl"].read_text(encoding="utf-8").strip().splitlines()
            assert any("no common date" in json.loads(l).get("error", "") for l in lines if l)

    def test_fund_missing_cum_return_logged(self, tmp_path: Path) -> None:
        """边界：adjusted 存在、cum_return 缺失，记录 数据是否缺失=是"""
        base = tmp_path / "base"
        adjusted = base / "fund_adjusted_nav_by_code"
        cum = base / "fund_cum_return_by_code"
        adjusted.mkdir(parents=True)
        cum.mkdir(parents=True)
        _write_csv(adjusted / "000001.csv", pd.DataFrame({"净值日期": ["2024-01-01"], "复权净值": [1.0]}))
        out_dir = tmp_path / "out"
        from v2.compare.compare_adjusted_nav_and_cum_return_window import (
            compare_adjusted_nav_and_cum_return_window,
        )
        result = compare_adjusted_nav_and_cum_return_window(
            base_dir=base,
            start_date="2024-01-01",
            end_date="2024-01-31",
            output_dir=out_dir,
        )
        summary = pd.read_csv(result["summary_csv"], dtype={"基金代码": str})
        assert summary.iloc[0]["数据是否缺失"] == "是"
        assert result["error_jsonl"].exists()
        rec = json.loads(result["error_jsonl"].read_text(encoding="utf-8").strip().splitlines()[0])
        assert "cum_return_missing" in rec.get("error", "")


# ========== filter_funds_for_next_step (v2) ==========


class TestFilterFundsForNextStepV2:
    """Filter v2 模块测试（支持 end-date，规则4/5 使用 [start_date,end_date]）"""

    def test_all_rules_pass(self, tmp_path: Path) -> None:
        """正常：规则1-5全通过"""
        base = tmp_path / "fund_etl"
        base.mkdir(parents=True)
        compare_details = tmp_path / "compare" / "details"
        integrity_details = tmp_path / "integrity" / "details"
        compare_details.mkdir(parents=True)
        integrity_details.mkdir(parents=True)

        _write_csv(base / "fund_purchase.csv", pd.DataFrame({"基金代码": ["000001"]}))
        _write_csv(base / "fund_overview.csv", pd.DataFrame({"基金代码": ["000001"]}))
        (base / "fund_nav_by_code").mkdir()
        (base / "fund_adjusted_nav_by_code").mkdir()
        _write_csv(base / "fund_nav_by_code" / "000001.csv", pd.DataFrame({"基金代码": ["000001"], "净值日期": ["2025-01-02"], "单位净值": [1.0]}))
        _write_csv(base / "fund_adjusted_nav_by_code" / "000001.csv", pd.DataFrame({"基金代码": ["000001"], "净值日期": ["2025-01-02"], "复权净值": [1.0]}))

        _write_csv(
            compare_details / "000001.csv",
            pd.DataFrame({"期初日期": ["2025-01-02"], "期末日期": ["2025-01-03"], "本地远程收益率偏差": ["0.01"]}),
        )
        _write_csv(
            integrity_details / "000001_2025-01-01_2025-12-31.csv",
            pd.DataFrame({"交易日日期": ["2025-01-02"], "该日期数据是否存在": ["是"]}),
        )

        from v2.filters.filter_funds_for_next_step import filter_funds_for_next_step
        out = filter_funds_for_next_step(
            purchase_csv=base / "fund_purchase.csv",
            overview_csv=base / "fund_overview.csv",
            nav_dir=base / "fund_nav_by_code",
            adjusted_nav_dir=base / "fund_adjusted_nav_by_code",
            compare_details_dir=compare_details,
            integrity_details_dir=integrity_details,
            start_date="2025-01-01",
            end_date="2025-12-31",
            max_abs_deviation=0.02,
        )
        by_code = {r["基金编码"]: r for r in out.to_dict("records")}
        assert by_code["000001"]["是否过滤"] == "否"

    def test_rule4_deviation_at_threshold_filtered(self, tmp_path: Path) -> None:
        """边界：偏差绝对值等于 max_abs_deviation 应过滤（>=）"""
        base = tmp_path / "fund_etl"
        base.mkdir(parents=True)
        compare_details = tmp_path / "compare" / "details"
        integrity_details = tmp_path / "integrity" / "details"
        compare_details.mkdir(parents=True)
        integrity_details.mkdir(parents=True)

        _write_csv(base / "fund_purchase.csv", pd.DataFrame({"基金代码": ["000001"]}))
        _write_csv(base / "fund_overview.csv", pd.DataFrame({"基金代码": ["000001"]}))
        (base / "fund_nav_by_code").mkdir()
        (base / "fund_adjusted_nav_by_code").mkdir()
        _write_csv(base / "fund_nav_by_code" / "000001.csv", pd.DataFrame({"基金代码": ["000001"], "净值日期": ["2025-01-02"], "单位净值": [1.0]}))
        _write_csv(base / "fund_adjusted_nav_by_code" / "000001.csv", pd.DataFrame({"基金代码": ["000001"], "净值日期": ["2025-01-02"], "复权净值": [1.0]}))

        _write_csv(
            compare_details / "000001.csv",
            pd.DataFrame({"期初日期": ["2025-01-02"], "期末日期": ["2025-01-03"], "本地远程收益率偏差": ["0.02"]}),
        )
        _write_csv(
            integrity_details / "000001_2025-01-01_2025-12-31.csv",
            pd.DataFrame({"交易日日期": ["2025-01-02"], "该日期数据是否存在": ["是"]}),
        )

        from v2.filters.filter_funds_for_next_step import filter_funds_for_next_step
        out = filter_funds_for_next_step(
            purchase_csv=base / "fund_purchase.csv",
            overview_csv=base / "fund_overview.csv",
            nav_dir=base / "fund_nav_by_code",
            adjusted_nav_dir=base / "fund_adjusted_nav_by_code",
            compare_details_dir=compare_details,
            integrity_details_dir=integrity_details,
            start_date="2025-01-01",
            end_date="2025-12-31",
            max_abs_deviation=0.02,
        )
        by_code = {r["基金编码"]: r for r in out.to_dict("records")}
        assert by_code["000001"]["是否过滤"] == "是"
        assert "规则4" in by_code["000001"]["过滤原因"]

    def test_rule4_detail_outside_window_ignored(self, tmp_path: Path) -> None:
        """边界：compare 记录在区间外，规则4 不触发（区间内无超标记录则通过）"""
        base = tmp_path / "fund_etl"
        base.mkdir(parents=True)
        compare_details = tmp_path / "compare" / "details"
        integrity_details = tmp_path / "integrity" / "details"
        compare_details.mkdir(parents=True)
        integrity_details.mkdir(parents=True)

        _write_csv(base / "fund_purchase.csv", pd.DataFrame({"基金代码": ["000001"]}))
        _write_csv(base / "fund_overview.csv", pd.DataFrame({"基金代码": ["000001"]}))
        (base / "fund_nav_by_code").mkdir()
        (base / "fund_adjusted_nav_by_code").mkdir()
        _write_csv(base / "fund_nav_by_code" / "000001.csv", pd.DataFrame({"基金代码": ["000001"], "净值日期": ["2025-01-02"], "单位净值": [1.0]}))
        _write_csv(base / "fund_adjusted_nav_by_code" / "000001.csv", pd.DataFrame({"基金代码": ["000001"], "净值日期": ["2025-01-02"], "复权净值": [1.0]}))

        # 区间 2025-06-01~2025-12-31 内：只有 2025-06-02 的记录，偏差 0.01 合格
        _write_csv(
            compare_details / "000001.csv",
            pd.DataFrame({
                "期初日期": ["2022-01-02", "2025-06-02"],
                "期末日期": ["2022-01-03", "2025-06-03"],
                "本地远程收益率偏差": ["0.50", "0.01"],
            }),
        )
        _write_csv(
            integrity_details / "000001_2025-06-01_2025-12-31.csv",
            pd.DataFrame({"交易日日期": ["2025-06-02"], "该日期数据是否存在": ["是"]}),
        )

        from v2.filters.filter_funds_for_next_step import filter_funds_for_next_step
        out = filter_funds_for_next_step(
            purchase_csv=base / "fund_purchase.csv",
            overview_csv=base / "fund_overview.csv",
            nav_dir=base / "fund_nav_by_code",
            adjusted_nav_dir=base / "fund_adjusted_nav_by_code",
            compare_details_dir=compare_details,
            integrity_details_dir=integrity_details,
            start_date="2025-06-01",
            end_date="2025-12-31",
            max_abs_deviation=0.02,
        )
        by_code = {r["基金编码"]: r for r in out.to_dict("records")}
        assert by_code["000001"]["是否过滤"] == "否"

    def test_start_after_end_raises(self, tmp_path: Path) -> None:
        """异常：start_date > end_date"""
        base = tmp_path / "fund_etl"
        base.mkdir(parents=True)
        _write_csv(base / "fund_purchase.csv", pd.DataFrame({"基金代码": ["000001"]}))
        (base / "fund_nav_by_code").mkdir()
        (base / "fund_adjusted_nav_by_code").mkdir()
        compare_details = tmp_path / "compare" / "details"
        integrity_details = tmp_path / "integrity" / "details"
        compare_details.mkdir(parents=True)
        integrity_details.mkdir(parents=True)
        from v2.filters.filter_funds_for_next_step import filter_funds_for_next_step
        with pytest.raises(ValueError, match="start-date cannot be after end-date"):
            filter_funds_for_next_step(
                purchase_csv=base / "fund_purchase.csv",
                overview_csv=base / "fund_overview.csv",
                nav_dir=base / "fund_nav_by_code",
                adjusted_nav_dir=base / "fund_adjusted_nav_by_code",
                compare_details_dir=compare_details,
                integrity_details_dir=integrity_details,
                start_date="2025-12-31",
                end_date="2025-01-01",
            )

    def test_filter_extended_end_date_detail_in_range(self, tmp_path: Path) -> None:
        """边界：延伸 end_date（差异3）时，延伸区间内 compare 记录参与判定"""
        base = tmp_path / "fund_etl"
        base.mkdir(parents=True)
        compare_details = tmp_path / "compare" / "details"
        integrity_details = tmp_path / "integrity" / "details"
        compare_details.mkdir(parents=True)
        integrity_details.mkdir(parents=True)

        _write_csv(base / "fund_purchase.csv", pd.DataFrame({"基金代码": ["000001"]}))
        _write_csv(base / "fund_overview.csv", pd.DataFrame({"基金代码": ["000001"]}))
        (base / "fund_nav_by_code").mkdir()
        (base / "fund_adjusted_nav_by_code").mkdir()
        _write_csv(base / "fund_nav_by_code" / "000001.csv", pd.DataFrame({"基金代码": ["000001"], "净值日期": ["2025-01-02"], "单位净值": [1.0]}))
        _write_csv(base / "fund_adjusted_nav_by_code" / "000001.csv", pd.DataFrame({"基金代码": ["000001"], "净值日期": ["2025-01-02"], "复权净值": [1.0]}))

        # 延伸后 end_date=2026-01-15；compare 记录 2026-01-10~2026-01-15 在延伸区间内
        _write_csv(
            compare_details / "000001.csv",
            pd.DataFrame({
                "期初日期": ["2025-06-02", "2026-01-10"],
                "期末日期": ["2025-06-03", "2026-01-15"],
                "本地远程收益率偏差": ["0.01", "0.01"],
            }),
        )
        _write_csv(
            integrity_details / "000001_2025-01-01_2026-01-15.csv",
            pd.DataFrame({"交易日日期": ["2025-01-02", "2026-01-10"], "该日期数据是否存在": ["是", "是"]}),
        )

        from v2.filters.filter_funds_for_next_step import filter_funds_for_next_step
        out = filter_funds_for_next_step(
            purchase_csv=base / "fund_purchase.csv",
            overview_csv=base / "fund_overview.csv",
            nav_dir=base / "fund_nav_by_code",
            adjusted_nav_dir=base / "fund_adjusted_nav_by_code",
            compare_details_dir=compare_details,
            integrity_details_dir=integrity_details,
            start_date="2025-01-01",
            end_date="2026-01-15",
            max_abs_deviation=0.02,
        )
        by_code = {r["基金编码"]: r for r in out.to_dict("records")}
        assert by_code["000001"]["是否过滤"] == "否"

    def test_empty_purchase_returns_empty_df(self, tmp_path: Path) -> None:
        """边界：申购列表为空"""
        base = tmp_path / "fund_etl"
        base.mkdir(parents=True)
        _write_csv(base / "fund_purchase.csv", pd.DataFrame({"基金代码": []}))
        _write_csv(base / "fund_overview.csv", pd.DataFrame({"基金代码": []}))
        (base / "fund_nav_by_code").mkdir()
        (base / "fund_adjusted_nav_by_code").mkdir()
        compare_details = tmp_path / "compare" / "details"
        integrity_details = tmp_path / "integrity" / "details"
        compare_details.mkdir(parents=True)
        integrity_details.mkdir(parents=True)
        from v2.filters.filter_funds_for_next_step import filter_funds_for_next_step
        out = filter_funds_for_next_step(
            purchase_csv=base / "fund_purchase.csv",
            overview_csv=base / "fund_overview.csv",
            nav_dir=base / "fund_nav_by_code",
            adjusted_nav_dir=base / "fund_adjusted_nav_by_code",
            compare_details_dir=compare_details,
            integrity_details_dir=integrity_details,
            start_date="2025-01-01",
            end_date="2025-12-31",
        )
        assert len(out) == 0


# ========== prep_eligible_window ==========


class TestPrepEligibleWindow:
    """v2 预备 eligible 按窗口计算"""

    def _minimal_work_dir(
        self,
        tmp_path: Path,
        codes: list[str],
        *,
        gmbd_scale: float = 3.0,
        inc_date: str = "2010-01-01",
    ) -> Path:
        work = tmp_path / "work"
        work.mkdir(parents=True)
        _write_csv(work / "fund_purchase.csv", pd.DataFrame({"基金代码": codes}))
        _write_csv(
            work / "fund_fee_filtered.csv",
            pd.DataFrame({"基金编码": codes, "类型": ["A类30天"] * len(codes), "申购费率": ["0.1%"] * len(codes), "赎回费率": ["0%"] * len(codes)}),
        )
        _write_csv(
            work / "fund_cyrjg.csv",
            pd.DataFrame({"基金代码": codes, "日期": ["2024-01-01"] * len(codes), "机构持有比例": ["30%"] * len(codes)}),
        )
        _write_csv(
            work / "fund_gmbd.csv",
            pd.DataFrame({"基金代码": codes, "日期": ["2024-01-01"] * len(codes), "期末净资产（亿元）": [str(gmbd_scale)] * len(codes)}),
        )
        _write_csv(
            work / "fund_overview.csv",
            pd.DataFrame({"基金代码": codes, "成立日期/规模": [inc_date] * len(codes)}),
        )
        _write_csv(
            work / "fund_fee_structured.csv",
            pd.DataFrame({"基金编码": codes, "申购状态": ["开放申购"] * len(codes), "赎回状态": ["开放赎回"] * len(codes)}),
        )
        return work

    def test_normal_eligible_all_pass(self, tmp_path: Path) -> None:
        """正常：c1+b+e 全通过"""
        work = self._minimal_work_dir(tmp_path, ["000001", "000002"])
        from v2.filters.prep_eligible_window import run
        out = run(work_dir=work, start_date="2024-01-01", end_date="2024-12-31", output_path=tmp_path / "out" / "eligible.csv")
        result = pd.read_csv(out, dtype=str)
        assert set(result["基金代码"].str.zfill(6)) >= {"000001", "000002"}

    def test_start_after_end_raises(self, tmp_path: Path) -> None:
        """异常：start_date > end_date"""
        work = self._minimal_work_dir(tmp_path, ["000001"])
        from v2.filters.prep_eligible_window import run
        with pytest.raises(ValueError, match="start-date cannot be after end-date"):
            run(work_dir=work, start_date="2024-12-31", end_date="2024-01-01")

    def test_missing_purchase_raises(self, tmp_path: Path) -> None:
        """异常：缺少 fund_purchase.csv"""
        work = tmp_path / "work"
        work.mkdir()
        from v2.filters.prep_eligible_window import run
        with pytest.raises(FileNotFoundError, match="missing input"):
            run(work_dir=work, start_date="2024-01-01", end_date="2024-12-31")

    def test_missing_cyrjg_raises(self, tmp_path: Path) -> None:
        """异常：缺少 fund_cyrjg.csv"""
        work = self._minimal_work_dir(tmp_path, ["000001"])
        (work / "fund_cyrjg.csv").unlink()
        from v2.filters.prep_eligible_window import run
        with pytest.raises(FileNotFoundError, match="missing input"):
            run(work_dir=work, start_date="2024-01-01", end_date="2024-12-31")

    def test_missing_gmbd_raises(self, tmp_path: Path) -> None:
        """异常：缺少 fund_gmbd.csv"""
        work = self._minimal_work_dir(tmp_path, ["000001"])
        (work / "fund_gmbd.csv").unlink()
        from v2.filters.prep_eligible_window import run
        with pytest.raises(FileNotFoundError, match="missing input"):
            run(work_dir=work, start_date="2024-01-01", end_date="2024-12-31")

    def test_empty_result_when_all_filtered(self, tmp_path: Path) -> None:
        """边界：所有基金被过滤，输出空 csv"""
        work = self._minimal_work_dir(tmp_path, ["000099"], gmbd_scale=1.0)
        _write_csv(
            work / "fund_fee_filtered.csv",
            pd.DataFrame({"基金编码": ["000001"], "类型": ["A类30天"]}),
        )
        from v2.filters.prep_eligible_window import run
        out = run(work_dir=work, start_date="2024-01-01", end_date="2024-12-31", output_path=tmp_path / "eligible.csv")
        result = pd.read_csv(out, dtype=str)
        assert result.empty

    def test_custom_output_path(self, tmp_path: Path) -> None:
        """正常：指定 output_path 写入到非 work_dir"""
        work = self._minimal_work_dir(tmp_path, ["000001"])
        custom = tmp_path / "custom" / "my_eligible.csv"
        from v2.filters.prep_eligible_window import run
        out = run(work_dir=work, start_date="2024-01-01", end_date="2024-12-31", output_path=custom)
        assert out == custom.resolve()
        assert custom.exists()

    def test_rule_f_personnel_in_window_excluded(self, tmp_path: Path) -> None:
        """规则f：[end_date-1年, end_date] 内有人事变动记录的基金应排除"""
        work = self._minimal_work_dir(tmp_path, ["000001", "000002"])
        personnel_dir = tmp_path / "personnel"
        personnel_dir.mkdir(parents=True)
        # 000001: 2024-06-01 人事变动，在 [2023-12-31, 2024-12-31] 窗口内，应排除
        _write_csv(
            personnel_dir / "000001.csv",
            pd.DataFrame({
                "基金代码": ["000001"],
                "公告标题": ["基金经理调整"],
                "基金名称": ["测试基金"],
                "公告日期": ["2024-06-01"],
                "报告ID": ["AN1"],
            }),
        )
        # 000002: 无人事文件，应保留
        from v2.filters.prep_eligible_window import run
        out = run(
            work_dir=work,
            start_date="2024-01-01",
            end_date="2024-12-31",
            personnel_dir=personnel_dir,
            output_path=tmp_path / "eligible.csv",
        )
        result = pd.read_csv(out, dtype=str)
        codes = set(result["基金代码"].str.zfill(6))
        assert "000001" not in codes
        assert "000002" in codes

    def test_rule_f_personnel_outside_window_retained(self, tmp_path: Path) -> None:
        """规则f：人事变动在窗口外应保留"""
        work = self._minimal_work_dir(tmp_path, ["000001"])
        personnel_dir = tmp_path / "personnel"
        personnel_dir.mkdir(parents=True)
        # 2023-01-01 在 [2023-12-31, 2024-12-31] 窗口前，应保留
        _write_csv(
            personnel_dir / "000001.csv",
            pd.DataFrame({
                "基金代码": ["000001"],
                "公告标题": ["基金经理调整"],
                "公告日期": ["2023-01-01"],
                "报告ID": ["AN1"],
            }),
        )
        from v2.filters.prep_eligible_window import run
        out = run(
            work_dir=work,
            start_date="2024-01-01",
            end_date="2024-12-31",
            personnel_dir=personnel_dir,
            output_path=tmp_path / "eligible.csv",
        )
        result = pd.read_csv(out, dtype=str)
        assert len(result) >= 1
        assert "000001" in set(result["基金代码"].str.zfill(6))

    def test_rule_a_consecutive_over_60_excluded(self, tmp_path: Path) -> None:
        """规则a：[成立+2年,end_date] 内机构持仓连续两次>60% 应排除"""
        work = tmp_path / "work"
        work.mkdir(parents=True)
        codes = ["000001"]
        _write_csv(work / "fund_purchase.csv", pd.DataFrame({"基金代码": codes}))
        _write_csv(
            work / "fund_fee_filtered.csv",
            pd.DataFrame({"基金编码": codes, "类型": ["A类30天"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}),
        )
        # 成立 2020-01-01，成立+2年=2022-01-01；2022-06-01 与 2022-09-01 连续两次>60%
        _write_csv(
            work / "fund_cyrjg.csv",
            pd.DataFrame({
                "基金代码": codes * 3,
                "日期": ["2022-06-01", "2022-09-01", "2022-12-01"],
                "机构持有比例": ["65%", "70%", "30%"],
            }),
        )
        _write_csv(
            work / "fund_gmbd.csv",
            pd.DataFrame({"基金代码": codes, "日期": ["2024-01-01"], "期末净资产（亿元）": ["5"]}),
        )
        _write_csv(
            work / "fund_overview.csv",
            pd.DataFrame({"基金代码": codes, "成立日期/规模": ["2020-01-01"]}),
        )
        _write_csv(
            work / "fund_fee_structured.csv",
            pd.DataFrame({"基金编码": codes, "申购状态": ["开放申购"], "赎回状态": ["开放赎回"]}),
        )
        from v2.filters.prep_eligible_window import run
        out = run(work_dir=work, start_date="2024-01-01", end_date="2024-12-31", output_path=tmp_path / "eligible.csv")
        result = pd.read_csv(out, dtype=str)
        assert result.empty

    def test_rule_a_single_over_60_not_excluded(self, tmp_path: Path) -> None:
        """规则a：仅一次>60% 不满足连续两次，不排除"""
        work = tmp_path / "work"
        work.mkdir(parents=True)
        codes = ["000001"]
        _write_csv(work / "fund_purchase.csv", pd.DataFrame({"基金代码": codes}))
        _write_csv(
            work / "fund_fee_filtered.csv",
            pd.DataFrame({"基金编码": codes, "类型": ["A类30天"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}),
        )
        # 成立 2020-01-01；仅 2022-06-01 一次>60%，下次 2022-09-01 为 30%
        _write_csv(
            work / "fund_cyrjg.csv",
            pd.DataFrame({
                "基金代码": codes * 2,
                "日期": ["2022-06-01", "2022-09-01"],
                "机构持有比例": ["65%", "30%"],
            }),
        )
        _write_csv(
            work / "fund_gmbd.csv",
            pd.DataFrame({"基金代码": codes, "日期": ["2024-01-01"], "期末净资产（亿元）": ["5"]}),
        )
        _write_csv(
            work / "fund_overview.csv",
            pd.DataFrame({"基金代码": codes, "成立日期/规模": ["2020-01-01"]}),
        )
        _write_csv(
            work / "fund_fee_structured.csv",
            pd.DataFrame({"基金编码": codes, "申购状态": ["开放申购"], "赎回状态": ["开放赎回"]}),
        )
        from v2.filters.prep_eligible_window import run
        out = run(work_dir=work, start_date="2024-01-01", end_date="2024-12-31", output_path=tmp_path / "eligible.csv")
        result = pd.read_csv(out, dtype=str)
        assert len(result) >= 1

    def test_rule_b_latest_over_2e_retained(self, tmp_path: Path) -> None:
        """规则b：end_date 前最新一条规模>2亿 应保留"""
        work = tmp_path / "work"
        work.mkdir(parents=True)
        codes = ["000001"]
        _write_csv(work / "fund_purchase.csv", pd.DataFrame({"基金代码": codes}))
        _write_csv(
            work / "fund_fee_filtered.csv",
            pd.DataFrame({"基金编码": codes, "类型": ["A类30天"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}),
        )
        _write_csv(
            work / "fund_cyrjg.csv",
            pd.DataFrame({"基金代码": codes, "日期": ["2024-01-01"], "机构持有比例": ["30%"]}),
        )
        # 2024-06-01 规模 1 亿，2024-12-15 规模 3 亿；最新>2亿，应保留
        _write_csv(
            work / "fund_gmbd.csv",
            pd.DataFrame({
                "基金代码": codes * 2,
                "日期": ["2024-06-01", "2024-12-15"],
                "期末净资产（亿元）": ["1", "3"],
            }),
        )
        _write_csv(
            work / "fund_overview.csv",
            pd.DataFrame({"基金代码": codes, "成立日期/规模": ["2010-01-01"]}),
        )
        _write_csv(
            work / "fund_fee_structured.csv",
            pd.DataFrame({"基金编码": codes, "申购状态": ["开放申购"], "赎回状态": ["开放赎回"]}),
        )
        from v2.filters.prep_eligible_window import run
        out = run(work_dir=work, start_date="2024-01-01", end_date="2024-12-31", output_path=tmp_path / "eligible.csv")
        result = pd.read_csv(out, dtype=str)
        assert len(result) >= 1

    def test_rule_b_same_date_multiple_rows_takes_last(self, tmp_path: Path) -> None:
        """规则b：同日期多行时 groupby.last 取最后一条"""
        work = tmp_path / "work"
        work.mkdir(parents=True)
        codes = ["000001"]
        _write_csv(work / "fund_purchase.csv", pd.DataFrame({"基金代码": codes}))
        _write_csv(
            work / "fund_fee_filtered.csv",
            pd.DataFrame({"基金编码": codes, "类型": ["A类30天"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}),
        )
        _write_csv(
            work / "fund_cyrjg.csv",
            pd.DataFrame({"基金代码": codes, "日期": ["2024-12-15"], "机构持有比例": ["30%"]}),
        )
        # 同日期 2024-12-15 两行：3亿、1亿；last 取 1亿，应排除
        _write_csv(
            work / "fund_gmbd.csv",
            pd.DataFrame({
                "基金代码": codes * 2,
                "日期": ["2024-12-15", "2024-12-15"],
                "期末净资产（亿元）": ["3", "1"],
            }),
        )
        _write_csv(
            work / "fund_overview.csv",
            pd.DataFrame({"基金代码": codes, "成立日期/规模": ["2010-01-01"]}),
        )
        _write_csv(
            work / "fund_fee_structured.csv",
            pd.DataFrame({"基金编码": codes, "申购状态": ["开放申购"], "赎回状态": ["开放赎回"]}),
        )
        from v2.filters.prep_eligible_window import run
        out = run(work_dir=work, start_date="2024-01-01", end_date="2024-12-31", output_path=tmp_path / "eligible.csv")
        result = pd.read_csv(out, dtype=str)
        assert result.empty

    def test_rule_b_latest_scale_before_end_date(self, tmp_path: Path) -> None:
        """规则b：仅用 end_date 前最新一条规模判定；窗口内曾>2亿但最新<=2亿应排除"""
        work = tmp_path / "work"
        work.mkdir(parents=True)
        codes = ["000001"]
        _write_csv(work / "fund_purchase.csv", pd.DataFrame({"基金代码": codes}))
        _write_csv(
            work / "fund_fee_filtered.csv",
            pd.DataFrame({"基金编码": codes, "类型": ["A类30天"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}),
        )
        _write_csv(
            work / "fund_cyrjg.csv",
            pd.DataFrame({"基金代码": codes, "日期": ["2024-01-01"], "机构持有比例": ["30%"]}),
        )
        # 2024-06-01 规模 3 亿，2024-12-15 规模 1 亿；end_date=2024-12-31 前最新为 1 亿，应排除
        _write_csv(
            work / "fund_gmbd.csv",
            pd.DataFrame({
                "基金代码": codes * 2,
                "日期": ["2024-06-01", "2024-12-15"],
                "期末净资产（亿元）": ["3", "1"],
            }),
        )
        _write_csv(
            work / "fund_overview.csv",
            pd.DataFrame({"基金代码": codes, "成立日期/规模": ["2010-01-01"]}),
        )
        _write_csv(
            work / "fund_fee_structured.csv",
            pd.DataFrame({"基金编码": codes, "申购状态": ["开放申购"], "赎回状态": ["开放赎回"]}),
        )
        from v2.filters.prep_eligible_window import run
        out = run(work_dir=work, start_date="2024-01-01", end_date="2024-12-31", output_path=tmp_path / "eligible.csv")
        result = pd.read_csv(out, dtype=str)
        assert result.empty

    def test_rule_a_exactly_60_not_excluded(self, tmp_path: Path) -> None:
        """边界：恰好60% 不算>60%，不排除"""
        work = tmp_path / "work"
        work.mkdir(parents=True)
        codes = ["000001"]
        _write_csv(work / "fund_purchase.csv", pd.DataFrame({"基金代码": codes}))
        _write_csv(
            work / "fund_fee_filtered.csv",
            pd.DataFrame({"基金编码": codes, "类型": ["A类30天"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}),
        )
        # 连续两次 60%、60% —— 60 不满足 >60，不排除
        _write_csv(
            work / "fund_cyrjg.csv",
            pd.DataFrame({
                "基金代码": codes * 2,
                "日期": ["2022-06-01", "2022-09-01"],
                "机构持有比例": ["60%", "60%"],
            }),
        )
        _write_csv(
            work / "fund_gmbd.csv",
            pd.DataFrame({"基金代码": codes, "日期": ["2024-01-01"], "期末净资产（亿元）": ["5"]}),
        )
        _write_csv(
            work / "fund_overview.csv",
            pd.DataFrame({"基金代码": codes, "成立日期/规模": ["2020-01-01"]}),
        )
        _write_csv(
            work / "fund_fee_structured.csv",
            pd.DataFrame({"基金编码": codes, "申购状态": ["开放申购"], "赎回状态": ["开放赎回"]}),
        )
        from v2.filters.prep_eligible_window import run
        out = run(work_dir=work, start_date="2024-01-01", end_date="2024-12-31", output_path=tmp_path / "eligible.csv")
        result = pd.read_csv(out, dtype=str)
        assert len(result) >= 1

    def test_rule_a_consecutive_before_start_ts_excluded(self, tmp_path: Path) -> None:
        """边界：[成立+2年, start_ts) 内连续两次>60% 也应排除（差异1：不再忽略该区间）"""
        work = tmp_path / "work"
        work.mkdir(parents=True)
        codes = ["000001"]
        _write_csv(work / "fund_purchase.csv", pd.DataFrame({"基金代码": codes}))
        _write_csv(
            work / "fund_fee_filtered.csv",
            pd.DataFrame({"基金编码": codes, "类型": ["A类30天"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}),
        )
        # 成立 2020-01-01，start=2024-01-01；连续两次>60% 仅在 2023-06/09（start 前）发生
        _write_csv(
            work / "fund_cyrjg.csv",
            pd.DataFrame({
                "基金代码": codes * 2,
                "日期": ["2023-06-01", "2023-09-01"],
                "机构持有比例": ["65%", "70%"],
            }),
        )
        _write_csv(
            work / "fund_gmbd.csv",
            pd.DataFrame({"基金代码": codes, "日期": ["2024-01-01"], "期末净资产（亿元）": ["5"]}),
        )
        _write_csv(
            work / "fund_overview.csv",
            pd.DataFrame({"基金代码": codes, "成立日期/规模": ["2020-01-01"]}),
        )
        _write_csv(
            work / "fund_fee_structured.csv",
            pd.DataFrame({"基金编码": codes, "申购状态": ["开放申购"], "赎回状态": ["开放赎回"]}),
        )
        from v2.filters.prep_eligible_window import run
        out = run(work_dir=work, start_date="2024-01-01", end_date="2024-12-31", output_path=tmp_path / "eligible.csv")
        result = pd.read_csv(out, dtype=str)
        assert result.empty

    def test_rule_b_scale_exactly_2_excluded(self, tmp_path: Path) -> None:
        """边界：最新规模恰好2亿 不保留（>2亿才保留）"""
        work = tmp_path / "work"
        work.mkdir(parents=True)
        codes = ["000001"]
        _write_csv(work / "fund_purchase.csv", pd.DataFrame({"基金代码": codes}))
        _write_csv(
            work / "fund_fee_filtered.csv",
            pd.DataFrame({"基金编码": codes, "类型": ["A类30天"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}),
        )
        _write_csv(
            work / "fund_cyrjg.csv",
            pd.DataFrame({"基金代码": codes, "日期": ["2024-01-01"], "机构持有比例": ["30%"]}),
        )
        _write_csv(
            work / "fund_gmbd.csv",
            pd.DataFrame({"基金代码": codes, "日期": ["2024-12-15"], "期末净资产（亿元）": ["2"]}),
        )
        _write_csv(
            work / "fund_overview.csv",
            pd.DataFrame({"基金代码": codes, "成立日期/规模": ["2010-01-01"]}),
        )
        _write_csv(
            work / "fund_fee_structured.csv",
            pd.DataFrame({"基金编码": codes, "申购状态": ["开放申购"], "赎回状态": ["开放赎回"]}),
        )
        from v2.filters.prep_eligible_window import run
        out = run(work_dir=work, start_date="2024-01-01", end_date="2024-12-31", output_path=tmp_path / "eligible.csv")
        result = pd.read_csv(out, dtype=str)
        assert result.empty

    def test_rule_b_no_scale_before_end_excluded(self, tmp_path: Path) -> None:
        """边界：end_date 前无规模数据应排除"""
        work = tmp_path / "work"
        work.mkdir(parents=True)
        codes = ["000001"]
        _write_csv(work / "fund_purchase.csv", pd.DataFrame({"基金代码": codes}))
        _write_csv(
            work / "fund_fee_filtered.csv",
            pd.DataFrame({"基金编码": codes, "类型": ["A类30天"], "申购费率": ["0.1%"], "赎回费率": ["0%"]}),
        )
        _write_csv(
            work / "fund_cyrjg.csv",
            pd.DataFrame({"基金代码": codes, "日期": ["2024-01-01"], "机构持有比例": ["30%"]}),
        )
        # 规模数据均在 end_date 之后
        _write_csv(
            work / "fund_gmbd.csv",
            pd.DataFrame({"基金代码": codes, "日期": ["2025-01-15"], "期末净资产（亿元）": ["5"]}),
        )
        _write_csv(
            work / "fund_overview.csv",
            pd.DataFrame({"基金代码": codes, "成立日期/规模": ["2010-01-01"]}),
        )
        _write_csv(
            work / "fund_fee_structured.csv",
            pd.DataFrame({"基金编码": codes, "申购状态": ["开放申购"], "赎回状态": ["开放赎回"]}),
        )
        from v2.filters.prep_eligible_window import run
        out = run(work_dir=work, start_date="2024-01-01", end_date="2024-12-31", output_path=tmp_path / "eligible.csv")
        result = pd.read_csv(out, dtype=str)
        assert result.empty
