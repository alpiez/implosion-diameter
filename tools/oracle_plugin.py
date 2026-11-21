"""pytest plugin used by tools/oracle.py: record every failing nodeid.

A test fails if any of its reports (setup/call/teardown, or any subtest report)
failed -- this covers pytest 9's PASSED-beside-SUBFAILED case, timeouts and
crashed xdist workers. Failed collection reports are recorded separately (the
whole file counts as failing). Results go to $ORACLE_OUT as JSON.
"""

import json
import os

_failed: set[str] = set()
_seen: set[str] = set()
_collect_errors: list[str] = []
_why: dict[str, str] = {}


def pytest_runtest_logreport(report):
    _seen.add(report.nodeid)
    if report.failed:
        _failed.add(report.nodeid)
        if report.nodeid not in _why:
            _why[report.nodeid] = f"[{report.when}] " + (report.longreprtext or "")[-1500:]


def pytest_collectreport(report):
    if report.failed:
        _collect_errors.append(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "workerinput"):  # xdist worker: controller reports
        return
    out = os.environ.get("ORACLE_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(
                {
                    "failed": sorted(_failed),
                    "seen": sorted(_seen),
                    "collect_errors": sorted(set(_collect_errors)),
                    "exitstatus": int(exitstatus),
                    "why": _why,
                },
                f,
            )
