from pathlib import Path

from optuna.storages import JournalStorage
from optuna.storages.journal import (
    JournalFileBackend,
    JournalFileOpenLock,
)


def create_journal_storage(journal_path: Path) -> JournalStorage:
    """Open the journal shared by the optimization processes."""
    resolved_path: Path = journal_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    file_path: str = str(resolved_path)
    lock: JournalFileOpenLock = JournalFileOpenLock(file_path)

    backend: JournalFileBackend = JournalFileBackend(
        file_path=file_path,
        lock_obj=lock,
    )

    return JournalStorage(backend)
