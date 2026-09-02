
from pathlib import Path
import subprocess
import time


class ProcessManager:

    def __init__(
        self,
        path_exe: Path,
    ) -> None:
        self.path_exe = path_exe
        self.process: subprocess.Popen | None = None
        self.executable_name = path_exe.name

    def start_process(self) -> None:
        if self.is_running():
            raise RuntimeError(
                f"{self.path_exe.name} is already running."
            )

        try:
            self.process = subprocess.Popen([str(self.path_exe)])
            self.wait_until_running(self.executable_name)

        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{self.executable_name} not found. Please install it and try again."
            ) from exc

        except TimeoutError as exc:
            raise TimeoutError(
                f"{self.executable_name} was found, but did not start within the expected time."
            ) from exc

        except PermissionError as exc:
            raise PermissionError(
                f"{self.executable_name} was found, but Python has no permission to start it."
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"{self.executable_name} could not be started due to an operating system error."
            ) from exc


    def stop_process(self) -> None:

        if self.process is None:
            return

        if self.process.poll() is None:
            subprocess.run(
                [
                    "taskkill",
                    "/F",
                    "/T",
                    "/PID",
                    str(self.process.pid),
                ],
                capture_output=True,
                text=True
            )

        self.process = None



    def is_running(self):

        tasklist = subprocess.run(
            ["tasklist"],
            capture_output=True,
            text=True
        )

        if Path(self.executable_name).name.lower() in tasklist.stdout.lower():
            return True
        else:
            return False


    def wait_until_running(self, executable_name, timeout_seconds: float = 30.0):

        end_time = time.time() + timeout_seconds

        while time.time() < end_time:
            if self.is_running():
                return

            time.sleep(0.5)

        raise TimeoutError(f"Process '{executable_name}' did not start within {timeout_seconds} seconds.")