import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import pytest


class _ProcessHandle(Protocol):
    def __int__(self) -> int: ...

    def close(self) -> None: ...


@pytest.mark.skipif(os.name != "nt", reason="Windows process protection")
def test_worker_exit_kills_child_but_preserves_unrelated_process(
    tmp_path: Path,
) -> None:
    import win32api
    import win32con
    import win32event

    project_root: Path = Path(__file__).resolve().parents[1]
    ready_path: Path = tmp_path / "child_pid.txt"

    worker_code: str = (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "from src.modules.process.worker_job import "
        "protect_worker_process_tree\n"
        "protect_worker_process_tree()\n"
        "child = subprocess.Popen([\n"
        "    sys.executable, '-c', 'import time; time.sleep(60)'\n"
        "])\n"
        "Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')\n"
        "time.sleep(60)\n"
    )

    open_process: Callable[[int, bool, int], _ProcessHandle] = cast(
        Callable[[int, bool, int], _ProcessHandle],
        cast(object, win32api.OpenProcess),
    )

    child_handle: _ProcessHandle | None = None
    unrelated: subprocess.Popen[bytes] | None = None

    worker: subprocess.Popen[bytes] = subprocess.Popen(
        [sys.executable, "-c", worker_code, str(ready_path)],
        cwd=project_root,
    )

    try:
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
        )

        deadline: float = time.monotonic() + 15.0
        child_pid: int | None = None

        while time.monotonic() < deadline:
            if worker.poll() is not None:
                pytest.fail(f"Worker exited before readiness: {worker.returncode}")

            if ready_path.exists():
                content: str = ready_path.read_text(encoding="utf-8").strip()
                if content:
                    child_pid = int(content)
                    break

            time.sleep(0.05)

        assert child_pid is not None, "Worker did not report its child PID."

        child_handle = open_process(
            win32con.SYNCHRONIZE | win32con.PROCESS_TERMINATE,
            False,
            child_pid,
        )

        assert (
            win32event.WaitForSingleObject(int(child_handle), 0)
            == win32event.WAIT_TIMEOUT
        )

        worker.kill()
        worker.wait(timeout=5)

        assert (
            win32event.WaitForSingleObject(int(child_handle), 5000)
            == win32event.WAIT_OBJECT_0
        )

        assert unrelated.poll() is None

    finally:
        try:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=5)
        finally:
            try:
                if child_handle is not None:
                    try:
                        if (
                            win32event.WaitForSingleObject(int(child_handle), 0)
                            == win32event.WAIT_TIMEOUT
                        ):
                            win32api.TerminateProcess(int(child_handle), 1)
                            win32event.WaitForSingleObject(int(child_handle), 5000)
                    finally:
                        child_handle.close()
            finally:
                if unrelated is not None:
                    if unrelated.poll() is None:
                        unrelated.kill()
                    unrelated.wait(timeout=5)
