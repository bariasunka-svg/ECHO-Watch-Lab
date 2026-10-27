# SPDX-License-Identifier: AGPL-3.0-only
"""Test-only admission gate.

Unit/regression tests exercise policy LOGIC on synthetic pixels; they must not depend on (or silently stand in for)
the frozen runtime calibration. Every test runs with an explicit, unqualified TEST gate unless it builds its own.
"""
import pytest

from echo_watch import matching

TEST_GATE = matching.AdmissionGate(matching.DESCRIPTOR_VERSION, 0.75, 0.45, "TEST-GATE", False, "unit tests only")


@pytest.fixture(autouse=True)
def _test_gate(monkeypatch):
    monkeypatch.setattr(matching, "_GATE", TEST_GATE)
    yield
