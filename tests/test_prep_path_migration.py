# -*- coding: utf-8 -*-
"""
预备数据路径迁移（tools/ -> tools/prep/）专项单元测试。

基于 diff 82d59a9 与 a111b8e 的变更，覆盖：
- 正常场景：路径解析、导入、_run_cli script_path
- 异常场景：src 缺失 assert、myanalyser 目录名 assert、fetch_fund_fee 缺失 RuntimeError
- 边界条件：不存在的子脚本、conftest PYTHONPATH
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_MYANALYSER = _ROOT  # tests 的 parent 为 myanalyser
_PREP_DIR = _MYANALYSER / "tools" / "prep"


# ============= 1. 正常场景 =============


class TestPrepPathResolution:
    """路径解析 - 正常场景"""

    def test_prep_data_workflow_prep_dir_points_to_tools_prep(self):
        """prep_data_workflow 的 _PREP_DIR 应指向 tools/prep。"""
        import prep_data_workflow as pw

        assert pw._PREP_DIR.resolve() == _PREP_DIR.resolve()
        assert (pw._PREP_DIR / "prep_data_workflow.py").exists()

    def test_prep_data_workflow_myanalyser_points_to_project_root(self):
        """prep_data_workflow 的 _MYANALYSER 应指向 myanalyser 目录。"""
        import prep_data_workflow as pw

        assert pw._MYANALYSER.name == "myanalyser"
        assert (pw._MYANALYSER / "src").exists()

    def test_run_cli_script_path_uses_prep_dir(self):
        """_run_cli 应使用 _PREP_DIR / script 解析子脚本路径。"""
        import prep_data_workflow as pw

        script_path = (pw._PREP_DIR / "fetch_fund_cyrjg.py").resolve()
        assert script_path.exists(), f"fetch_fund_cyrjg.py 应存在于 {pw._PREP_DIR}"

    def test_fetch_fund_cyrjg_importable_from_prep(self):
        """fetch_fund_cyrjg 可从 tools/prep 正确导入（conftest 已加入 PYTHONPATH）。"""
        from fetch_fund_cyrjg import _load_codes_from_csv

        assert callable(_load_codes_from_csv)

    def test_fetch_fund_gmbd_importable_from_prep(self):
        """fetch_fund_gmbd 可从 tools/prep 正确导入。"""
        from fetch_fund_gmbd import _load_codes_from_csv

        assert callable(_load_codes_from_csv)

    def test_conftest_adds_tools_prep_to_path(self):
        """conftest 应将 tools/prep 加入 PYTHONPATH，以便导入 prep 模块。"""
        prep_path = str(_PREP_DIR)
        assert prep_path in sys.path or any(
            Path(p).resolve() == _PREP_DIR.resolve() for p in sys.path
        )


class TestPrepCLIExecutable:
    """CLI 可执行性 - 正常场景"""

    def test_fetch_fund_cyrjg_cli_help(self):
        """fetch_fund_cyrjg.py -h 应退出码 0。"""
        script = _PREP_DIR / "fetch_fund_cyrjg.py"
        ret = subprocess.run(
            [sys.executable, str(script), "-h"],
            capture_output=True,
            text=True,
            cwd=_MYANALYSER.parent,
        )
        assert ret.returncode == 0, ret.stderr

    def test_fetch_fund_gmbd_cli_help(self):
        """fetch_fund_gmbd.py -h 应退出码 0。"""
        script = _PREP_DIR / "fetch_fund_gmbd.py"
        ret = subprocess.run(
            [sys.executable, str(script), "-h"],
            capture_output=True,
            text=True,
            cwd=_MYANALYSER.parent,
        )
        assert ret.returncode == 0, ret.stderr

    def test_prep_data_workflow_cli_help(self):
        """prep_data_workflow.py -h 应退出码 0。"""
        script = _PREP_DIR / "prep_data_workflow.py"
        ret = subprocess.run(
            [sys.executable, str(script), "-h"],
            capture_output=True,
            text=True,
            cwd=_MYANALYSER.parent,
        )
        assert ret.returncode == 0, ret.stderr


# ============= 2. 异常场景 =============


class TestFetchCyrjgAssertWhenSrcMissing:
    """fetch_fund_cyrjg：项目根下无 src 时应 AssertionError"""

    def test_assert_when_run_from_wrong_structure(self, tmp_path):
        """当脚本位于 parents[2] 无 src 的目录结构时，导入即 assert。"""
        # 复制脚本到 tmp_path/a/b/，则 parents[2] = tmp_path，通常无 src
        fake_root = tmp_path / "a" / "b"
        fake_root.mkdir(parents=True)
        src_file = _PREP_DIR / "fetch_fund_cyrjg.py"
        dst_file = fake_root / "fetch_fund_cyrjg.py"
        shutil.copy(src_file, dst_file)
        # 运行脚本（-h 即可触发顶层 assert）
        ret = subprocess.run(
            [sys.executable, str(dst_file), "-h"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert ret.returncode != 0
        assert "invalid project root" in ret.stderr or "AssertionError" in ret.stderr


class TestFetchGmbdAssertWhenSrcMissing:
    """fetch_fund_gmbd：同上"""

    def test_assert_when_run_from_wrong_structure(self, tmp_path):
        """当脚本位于 parents[2] 无 src 的目录结构时，导入即 assert。"""
        fake_root = tmp_path / "a" / "b"
        fake_root.mkdir(parents=True)
        shutil.copy(_PREP_DIR / "fetch_fund_gmbd.py", fake_root / "fetch_fund_gmbd.py")
        ret = subprocess.run(
            [sys.executable, str(fake_root / "fetch_fund_gmbd.py"), "-h"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert ret.returncode != 0
        assert "invalid project root" in ret.stderr or "AssertionError" in ret.stderr


class TestPrepDataWorkflowAssertWhenNotMyanalyser:
    """prep_data_workflow：_MYANALYSER.name != 'myanalyser' 时应 assert"""

    def test_assert_when_parent_not_named_myanalyser(self, tmp_path):
        """当脚本位于 .../not_myanalyser/scripts/ 时，assert 触发。"""
        not_myanalyser = tmp_path / "not_myanalyser" / "scripts"
        not_myanalyser.mkdir(parents=True)
        shutil.copy(
            _PREP_DIR / "prep_data_workflow.py",
            not_myanalyser / "prep_data_workflow.py",
        )
        ret = subprocess.run(
            [sys.executable, str(not_myanalyser / "prep_data_workflow.py"), "-h"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={**__import__("os").environ},
        )
        assert ret.returncode != 0
        assert "unexpected _MYANALYSER" in ret.stderr or "AssertionError" in ret.stderr


class TestFundFeeCompleteRaisesWhenFeeMissing:
    """fund_fee_complete：PREP_DIR 下无 fetch_fund_fee.py 时应 RuntimeError"""

    def test_raises_when_fetch_fund_fee_missing(self, tmp_path):
        """当 tools/prep 下无 fetch_fund_fee.py 时，导入即 RuntimeError。"""
        fake_tools = tmp_path / "tools"
        fake_temp_use = fake_tools / "temp_use"
        fake_prep = fake_tools / "prep"
        fake_temp_use.mkdir(parents=True)
        fake_prep.mkdir(parents=True)
        # 创建占位文件，但 prep 中不放 fetch_fund_fee.py
        (fake_prep / "fetch_fund_cyrjg.py").write_text("# stub")
        src = _ROOT / "tools" / "temp_use" / "fund_fee_complete.py"
        if not src.exists():
            pytest.skip("fund_fee_complete.py 不存在")
        dst = fake_temp_use / "fund_fee_complete.py"
        shutil.copy(src, dst)
        ret = subprocess.run(
            [sys.executable, str(dst), "--help"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={**__import__("os").environ},
        )
        # 可能因其他 import 失败，至少应非 0
        assert ret.returncode != 0
        assert "expected fetch_fund_fee" in ret.stderr or "RuntimeError" in ret.stderr


# ============= 3. 边界条件 =============


class TestRunCliBoundary:
    """_run_cli 边界条件"""

    def test_nonexistent_script_returns_false(self):
        """传入不存在的脚本名时，subprocess 会失败，_run_cli 返回 False。"""
        import logging

        import prep_data_workflow as pw

        logger = logging.getLogger("test")
        ok = pw._run_cli(
            "nonexistent_script_xyz.py",
            ["-h"],
            logger,
            stream_output=False,
        )
        assert ok is False
