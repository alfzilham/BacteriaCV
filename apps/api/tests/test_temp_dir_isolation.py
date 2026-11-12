"""Guard tests for the per-process isolation of the shared pytest temp folder.

Two pytest processes running in this repository at the same time used to fight
over one directory named ``.pytest_tmp``. The autouse cleanup fixture removed
that directory after every test, so whichever process finished a test first
deleted the folder the other one was still reading from. The failures surfaced
as ``FileNotFoundError`` on a species directory and as ``ValueError`` from
``_relative_posix`` reporting the image as outside the project root.

The folder name now carries the process id, which is the only thing that
actually differs between two concurrent processes. The tests below exist so that
a silent revert cannot bring the collision back.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from bacteriacv.paths import PROJECT_ROOT

# The concurrency probe runs this whole file, not the suite. Every test in it
# builds its images inside the shared temp folder, so two processes running it
# at the same time genuinely overlap on that folder. Targeting one file also
# keeps the probe from recursing into itself.
_PROBE_TARGET = "tests/test_build_index.py"


def test_temp_dir_name_carries_the_running_pid(project_tmp_dir: Path) -> None:
    """The folder name must be unique per process, otherwise runs collide."""
    expected = f".pytest_tmp_{os.getpid()}"
    assert project_tmp_dir.name == expected, project_tmp_dir.name

    _, _, pid_part = project_tmp_dir.name.partition(".pytest_tmp_")
    assert pid_part.isdigit(), pid_part
    assert pid_part == str(os.getpid())


def test_temp_dir_name_is_not_a_fixed_string(project_tmp_dir: Path) -> None:
    """A hardcoded name is the exact defect this change removed."""
    assert project_tmp_dir.name != ".pytest_tmp", project_tmp_dir.name
    assert "{{" not in project_tmp_dir.name, project_tmp_dir.name


def test_two_concurrent_pytest_processes_do_not_collide() -> None:
    """Two pytest runs at the same time must both stay green.

    This is the regression that motivated the fix. With a single fixed folder
    name, one process removes the folder while the other is reading it, and the
    run fails with ``FileNotFoundError`` or a ``ValueError`` from
    ``_relative_posix``. The pid in the name gives each process its own
    directory, so both runs finish independently.

    Each process is retried up to three times. A collision needs the two runs to
    overlap on the folder, and scheduling decides that, so a single unlucky
    interleaving must not be reported as a regression.
    """
    results: dict[str, subprocess.CompletedProcess[str]] = {}

    def run(name: str) -> None:
        results[name] = subprocess.run(
            [
                sys.executable, "-m", "pytest", _PROBE_TARGET,
                "-q", "--no-header", "-p", "no:cacheprovider",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )

    attempts = []
    for _ in range(3):
        results.clear()
        threads = [threading.Thread(target=run, args=(name,)) for name in ("A", "B")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert set(results) == {"A", "B"}, "the probe processes did not both report"
        failures = {
            name: result
            for name, result in results.items()
            if result.returncode != 0
        }
        if not failures:
            break
        attempts.append(failures)

    assert not attempts, (
        "concurrent pytest processes kept failing, which means the temp folder "
        f"is still shared between processes. Failures per attempt: "
        f"{[sorted(f) for f in attempts]}"
    )