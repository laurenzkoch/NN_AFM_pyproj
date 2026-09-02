
from src.modules.process.process_management import ProcessManager
from src.config.config_paths import *


class EDUVPNProcessor(ProcessManager):

    def __init__(self) -> None:
        super().__init__(
            path_exe=PATH_eduVPN_EXE
        )

    def __enter__(self) -> "EDUVPNProcessor":
        print("-----------Starting eduVPN----------")

        if self.is_running():
            self.stop_process()

        self.start_process()

        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ) -> None:
        print("-----------Closing eduVPN----------")
        self.stop_process()