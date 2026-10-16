import logging
from types import TracebackType
from typing import Self

from src.config.config_paths import PATH_eduVPN_EXE
from src.modules.process.process_management import ProcessManager

logger = logging.getLogger(__name__)


class EDUVPNProcessor(ProcessManager):
    def __init__(self) -> None:
        super().__init__(path_exe=PATH_eduVPN_EXE)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self) -> Self:
        if self.is_running():
            logger.info("Using already running eduVPN")
            return self

        logger.info("Starting eduVPN")
        self.start_process()
        return self

    def stop(self) -> None:
        if self.process is None:
            logger.info("Leaving externally started eduVPN running")
            return

        logger.info("Closing eduVPN started by this processor")
        self.stop_process()
