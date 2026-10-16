import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessManager:
    def __init__(
        self,
        path_exe: Path,
    ) -> None:
        self.path_exe = path_exe
        self.process: subprocess.Popen[bytes] | None = None
        self.executable_name = path_exe.name

    def start_process(self) -> None:
        if self.is_running():
            logger.error(
                "%s is already running.",
                self.executable_name,
            )
            raise RuntimeError(f"{self.executable_name} is already running.")

        try:
            self.process = subprocess.Popen([str(self.path_exe)])
            self.wait_until_running(self.executable_name)

        except FileNotFoundError as exc:
            logger.exception(
                "%s not found. Please install it and try again.",
                self.executable_name,
            )
            raise FileNotFoundError(
                f"{self.executable_name} not found. Please install it and try again."
            ) from exc

        except TimeoutError as exc:
            logger.exception(
                "%s did not start within the expected time.",
                self.executable_name,
            )
            raise TimeoutError(
                f"{self.executable_name} did not start within the expected time."
            ) from exc

        except PermissionError as exc:
            logger.exception(
                "Python has no permission to start %s.",
                self.executable_name,
            )
            raise PermissionError(
                f"Python has no permission to start {self.executable_name}."
            ) from exc

        except OSError as exc:
            logger.exception(
                "%s could not be started due to an operating system error.",
                self.executable_name,
            )
            raise RuntimeError(
                f"{self.executable_name} could not be started "
                "due to an operating system error."
            ) from exc

    def stop_process(self) -> None:
        process: subprocess.Popen[bytes] | None = self.process

        if process is None:
            return

        if process.poll() is None:
            result: subprocess.CompletedProcess[str] = subprocess.run(
                [
                    "taskkill",
                    "/F",
                    "/T",
                    "/PID",
                    str(process.pid),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0 and process.poll() is None:
                details: str = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(
                    f"Could not stop process PID {process.pid}: {details}"
                )

        process.wait(timeout=10.0)
        self.process = None

    def is_running(self) -> bool:
        tasklist = subprocess.run(["tasklist"], capture_output=True, text=True)

        if Path(self.executable_name).name.lower() in tasklist.stdout.lower():
            return True
        else:
            return False

    def wait_until_running(
        self,
        executable_name: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        deadline: float = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            if self.is_running():
                return

            time.sleep(0.5)

        logger.error(
            "Process '%s' did not start within %s seconds.",
            executable_name,
            timeout_seconds,
        )
        raise TimeoutError(
            f"Process '{executable_name}' did not start "
            f"within {timeout_seconds} seconds."
        )
