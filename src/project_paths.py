from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sanitize_run_suffix(value: str | None) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def default_run_id(suffix: str | None = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cleaned_suffix = sanitize_run_suffix(suffix)
    if cleaned_suffix:
        return f"{ts}_{cleaned_suffix}"
    return ts
