import time

from modules.utilitis.process_managing import ProcessManager
from src.config import config_paths


class EDUVPNProcessor(ProcessManager):

    def __init__(self):
        super().__init__()

        print("-----------Starting eduVPN----------")
        if self.is_running(config_paths.PATH_eduVPN_EXE):
            self.stop_process(config_paths.PATH_eduVPN_EXE)
            time.sleep(2)
        self.vpn_process = self.start_process(config_paths.PATH_eduVPN_EXE)

    def __enter__(self) -> 'EDUVPNProcessor':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if self.vpn_process is not None:
                try:
                    self.stop_process(self.vpn_process)
                    time.sleep(2)
                finally:
                    self.vpn_process = None
        except Exception as exc:
            print(f"Error while closing eduVPN: {exc}")
        return None