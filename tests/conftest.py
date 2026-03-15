"""Pytest 配置：确保 myanalyser/src、myanalyser/tools、myanalyser/tools/prep 在 PYTHONPATH 中，供各测试模块导入。"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
for subdir in ("src", "tools", "tools/prep"):
    _p = _root / subdir
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
