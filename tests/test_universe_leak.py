"""Run the point-in-time universe check when the data cache is present.

It needs the built panels in cache/ (tens of GB, not in the repo), so it skips on a
fresh clone. Set SKIP_DATA_TESTS=1 to skip it even when the cache exists.
"""
import os
import subprocess
import sys

import pytest

from conftest import ROOT

NEEDED = ["cache/mp_close.parquet", "cache/mp_dv.parquet", "cache/char_R_fix.npy", "cache/char_idx_fix.pkl"]
missing = [p for p in NEEDED if not (ROOT / p).exists()]


@pytest.mark.skipif(bool(missing), reason=f"data cache not present: {missing}")
@pytest.mark.skipif(os.environ.get("SKIP_DATA_TESTS") == "1", reason="SKIP_DATA_TESTS=1")
def test_universe_is_point_in_time():
    result = subprocess.run([sys.executable, "universe_leak_check.py"], cwd=ROOT,
                            capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
