#!/usr/bin/env python3
"""verify step10 回测适配层入口。逻辑在 src/backtest_verify_e2e.py。"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from backtest_verify_e2e import main

if __name__ == "__main__":
    main()
