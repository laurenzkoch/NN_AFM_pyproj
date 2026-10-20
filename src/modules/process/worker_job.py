import os
from collections.abc import Callable
from typing import Protocol, cast
from uuid import uuid4


class _JobHandle(Protocol):
    def __int__(self) -> int: ...

    def close(self) -> None: ...


_worker_job_handle: _JobHandle | None = None


def protect_worker_process_tree() -> None:
    """Terminate this worker's child processes when the worker exits."""
    global _worker_job_handle

    if os.name != "nt":
        raise RuntimeError("Worker process protection requires Windows.")

    if _worker_job_handle is not None:
        return

    import win32api
    import win32job

    create_job: Callable[[None, str], _JobHandle] = cast(
        Callable[[None, str], _JobHandle],
        cast(object, win32job.CreateJobObject),
    )

    job_name: str = f"NN_AFM_worker_{os.getpid()}_{uuid4().hex}"
    job: _JobHandle = create_job(None, job_name)
    job_id: int = int(job)

    try:
        information = win32job.QueryInformationJobObject(
            job_id,
            win32job.JobObjectExtendedLimitInformation,
        )
        information["BasicLimitInformation"]["LimitFlags"] |= (
            win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )

        win32job.SetInformationJobObject(
            job_id,
            win32job.JobObjectExtendedLimitInformation,
            information,
        )
        win32job.AssignProcessToJobObject(
            job_id,
            int(win32api.GetCurrentProcess()),
        )
    except BaseException:
        job.close()
        raise

    _worker_job_handle = job