from pathlib import Path
import subprocess
import time


class ProcessManager:

    def __init__(self) -> None:
        pass

    def start_process(self, path_exe: Path):

        executable_name = path_exe.name

        if self.is_running(executable_name):
            print(f"{executable_name} is already running.")
            raise RuntimeError(f"{executable_name} is already running.")

        try:
            prozess = subprocess.Popen([str(path_exe)])
            self.wait_until_running(executable_name)
            return prozess

        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{executable_name} not found. Please install it and try again."
            ) from exc

        except TimeoutError as exc:
            raise TimeoutError(
                f"{executable_name} was found, but did not start within the expected time."
            ) from exc

        except PermissionError as exc:
            raise PermissionError(
                f"{executable_name} was found, but Python has no permission to start it."
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"{executable_name} could not be started due to an operating system error."
            ) from exc

    @staticmethod
    def is_running(executable_name):

        result = subprocess.run(
            ["tasklist"],
            capture_output=True,
            text=True
        )

        if Path(executable_name).name.lower() in result.stdout.lower():
            return True
        else:
            return False


    def wait_until_running(self, executable_name, timeout_seconds: float = 30.0):

        end_time = time.time() + timeout_seconds

        while time.time() < end_time:
            if self.is_running(executable_name):
                return

            time.sleep(0.5)

        raise TimeoutError(f"Process '{executable_name}' did not start within {timeout_seconds} seconds.")


    def stop_process(self, process, timeout_seconds: float = 10.0):

        if process is None:
            return

        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()