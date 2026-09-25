"""Without numba (e.g. Pyodide/WebAssembly) the same Somers' D code runs as plain Python.

The calculation is not replaced: the numba-compiled functions are the Python
functions, so without numba they run as written and must give the same counts.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

from fastwoe import metrics

CASES = """
import numpy as np
rng = np.random.default_rng(0)
cases = {
    "binary y, continuous x": (rng.integers(0, 2, 3000), rng.normal(size=3000)),
    "binary y, tied x": (rng.integers(0, 2, 3000), rng.integers(0, 7, 3000)),
    "proportion y, tied x": (rng.random(2000).round(2), rng.normal(size=2000).round(1)),
    "all x tied": (rng.integers(0, 2, 100), np.ones(100)),
    "all y tied": (np.ones(100), rng.normal(size=100)),
    "single row": (np.array([1.0]), np.array([0.3])),
}
w = rng.random(3000)
"""

COMPUTE = (
    CASES
    + """
import json
from fastwoe import metrics
out = {}
for name, (y, x) in cases.items():
    y, x = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)
    out[name] = [list(metrics._somers_yx_core(y, x)), list(metrics._somers_xy_core(y, x))]
y, x = (np.asarray(v, dtype=np.float64) for v in cases["binary y, tied x"])
out["weighted"] = list(metrics._somers_yx_weighted(y[:400], x[:400], w[:400]))
print(json.dumps({"numba": metrics._HAS_NUMBA, "out": out}, default=float))
"""
)


def run(code: str, block_numba: bool, platform: str = "") -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter, optionally with ``import numba`` failing."""
    prelude = "import sys; "
    if block_numba:
        prelude += "sys.modules['numba'] = None; "
    if platform:  # after numpy/pandas load, which inspect the real platform
        prelude += f"import numpy, pandas; sys.platform = {platform!r}; "
    return subprocess.run(
        [sys.executable, "-W", "always", "-c", prelude + code],
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.skipif(not metrics._HAS_NUMBA, reason="compares against numba")
def test_plain_python_gives_the_same_counts_as_numba():
    compiled = json.loads(run(COMPUTE, block_numba=False).stdout)
    plain = json.loads(run(COMPUTE, block_numba=True).stdout)
    assert compiled["numba"] is True and plain["numba"] is False
    assert plain["out"] == compiled["out"]  # exact, including the integer pair counts


def test_package_works_without_numba():
    out = run(
        "import numpy as np, pandas as pd; from fastwoe import FastWoe; "
        "from fastwoe.metrics import _HAS_NUMBA, somersd_yx; "
        "rng = np.random.default_rng(0); x = rng.normal(size=500); "
        "y = (rng.random(500) < 1 / (1 + np.exp(-x))).astype(int); "
        "m = FastWoe().fit(pd.DataFrame({'x': x}), y); "
        "print(_HAS_NUMBA, round(somersd_yx(y, m.predict_proba(pd.DataFrame({'x': x}))[:, 1]).statistic, 3))",
        block_numba=True,
    )
    has_numba, stat = out.stdout.split()
    assert has_numba == "False"
    assert 0.2 < float(stat) < 1
    assert "runs as plain Python" in out.stderr


def test_no_warning_under_webassembly():
    """Under Pyodide numba is absent by design, so importing says nothing."""
    out = run("import fastwoe.metrics", block_numba=True, platform="emscripten")
    assert "Numba not available" not in out.stderr


def test_auc_identity():
    """For a binary target Somers' D_{Y|X} is 2 AUC - 1, with or without numba."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 5000).astype(float)
    x = y + rng.normal(size=5000)
    assert metrics.somersd_yx(y, x).statistic == pytest.approx(
        2 * roc_auc_score(y, x) - 1, abs=1e-12
    )
