import subprocess
from pathlib import Path
import time


def is_running(process_name):
    result = subprocess.run(
        ["tasklist"],
        capture_output=True,
        text=True
    )
    return Path(process_name).name.lower() in result.stdout.lower()


def stop_process_by_name(process_name, timeout_seconds: float = 10.0):
    executable_name = Path(process_name).name
    end_time = time.time() + timeout_seconds

    while time.time() < end_time:
        if is_running(executable_name):
            subprocess.run(
                ["taskkill", "/F", "/T", "/IM", executable_name],
                capture_output=True,
                text=True
            )

        time.sleep(0.5)

    if is_running(executable_name):
        raise RuntimeError(f"Failed to stop process '{executable_name}'.")

def wait_until_running(executable_name, timeout_seconds: float = 30.0):

    end_time = time.time() + timeout_seconds

    while time.time() < end_time:
        if is_running(executable_name):
            return

        time.sleep(0.5)

    raise TimeoutError(f"Process '{executable_name}' did not start within {timeout_seconds} seconds.")


def start_process(process_name):
    process_path = Path(process_name)
    executable_name = process_path.name

    if is_running(executable_name):
        print(f"{executable_name} is already running. Restarting it...")
        stop_process_by_name(executable_name)

    if not process_path.exists():
        raise FileNotFoundError(
            f"Cannot start '{process_name}' because the executable was not found."
        )

    prozess = subprocess.Popen([str(process_path)])
    wait_until_running(executable_name)

    return prozess


def stop_process(process):
    if process is None:
        return

    if process.poll() is not None:
        process.wait()
        return

    process.terminate()

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()