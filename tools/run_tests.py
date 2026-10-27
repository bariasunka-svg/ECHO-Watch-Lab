# SPDX-License-Identifier: AGPL-3.0-only
"""Verification-only test runner: env/ui/python.exe -I tools/run_tests.py [pytest args].

The product runtime (env/ui) carries no test dependencies; pytest and its exact cached closure live in env/dev-site
(requirements/dev.lock) and are put on sys.path only for this process. Never constructs Tcl/Tk.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "env" / "dev-site"))
sys.path.insert(0, str(ROOT / "tests"))
os.chdir(ROOT)
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "cache" / "tmp")
import pytest  # noqa: E402

sys.exit(pytest.main(sys.argv[1:]))
