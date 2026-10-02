
from threading import Thread
from typing import Any
from queue import Queue
import queue
from datetime import datetime
from zoneinfo import ZoneInfo
import logging

from src.config.config_paths import *
from src.config.config_mcad_parameters import *

from src.modules.process.resources import determine_number_of_kernels
from src.modules.simulation.userinterface import MyUserInterface
from src.modules.simulation.generators.machine_generator import AFMMachineGenerator
from src.modules.simulation.run_simulation import MCADSimulation
from src.modules.data_managment.file_io import delete_folder
from src.modules.process.eduvpn import EDUVPNProcessor
from src.modules.data_managment.csv_io import create_csv_file


logger = logging.getLogger(__name__)


class SimulationPipeline:

    def __init__(
            self,
            num_kernel: int,
            num_base_designs: int,
            num_lam_designs: int,
            num_operation_points_per_dim: int,
            log_queue: Queue,
            autostart: bool = False,
            start_eduvpn: bool = True,
            gen_type: str = "v1",
            csv_path: str | None = None,
    ) -> None:

        if num_kernel < 1:
            raise ValueError("num_kernel must be >= 1")

        if num_base_designs < 1:
            raise ValueError("num_base_designs must be >= 1")

        if num_lam_designs < 1:
            raise ValueError("num_lam_designs must be >= 1")

        if num_operation_points_per_dim < 1:
            raise ValueError("num_operation_points_per_dim must be >= 1")


        self.autostart = autostart
        self.start_eduvpn = start_eduvpn
        self.ui: MyUserInterface | None = None
        self.simulation_thread: Thread | None = None
        self.simulation_exception: Any | None = None
        self.progress_queue: Queue[tuple[str, bool | None | Exception]] = Queue()
        self.eduvpn_processor: EDUVPNProcessor | None  = None
        self.gen_type: str = gen_type
        self.num_base_designs: int = num_base_designs
        self.num_lam_designs: int = num_lam_designs
        self.num_operation_points_per_dim: int = num_operation_points_per_dim
        self.num_datapoints: int = self._calculate_datapoints()
        self.simulated_points: int = 0
        self.dropped_points: int = 0
        self.existing_points: int = 0
        self.copied_points: int = 0
        self.csv_filepath: Path = self.create_csv_filepath() if csv_path is None else PATH_DATA_MCAD_RESULTS / csv_path
        self.num_workers: int = min(
            determine_number_of_kernels(num_kernel),
            self.num_datapoints
        )
        self.afmmachine_generator = AFMMachineGenerator(
            num_base_designs=num_base_designs,
            num_lam_designs=num_lam_designs,
            num_operation_points_per_dim=num_operation_points_per_dim,
            num_data_points=self.num_datapoints,
        )
        self.Simulation = MCADSimulation(
            point_generator=self.afmmachine_generator.get_generator(_type=self.gen_type),
            batch_size=self.num_operation_points_per_dim**3,
            progress_queue=self.progress_queue,
            csv_path=self.csv_filepath,
            num_workers=self.num_workers,
            log_queue=log_queue,
        )


    def start(self) -> None:
        if self.start_eduvpn:
            self.eduvpn_processor = EDUVPNProcessor().start()

        create_csv_file(
            path=self.csv_filepath,
            columns_list=CSV_PARAMETERS
        )

        self._initialize_simulation_thread()
        self._initialize_ui()
        self._initialize_progress_tracking()
        self._run_ui()

    def stop(self) -> None:
        delete_folder(PATH_MCAD_DATA / PATH_TEMP_FILES)
        if self.eduvpn_processor is not None:
            self.eduvpn_processor.stop()


    def __enter__(self) -> "SimulationPipeline":
        self.start()
        return self


    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


    def _calculate_datapoints(self) -> int:

        match self.gen_type:
            case "v1":
                return self.num_base_designs * self.num_lam_designs**3 * self.num_operation_points_per_dim**3
            case "v2":
                return self.num_base_designs * self.num_lam_designs * self.num_operation_points_per_dim**3
            case _:
                logger.exception("Invalid generator type.")
                raise ValueError("Invalid generator type.")


    def _initialize_progress_tracking(self) -> None:
        self._poll_queue()


    def _start_simulation(self) -> None:
        self.progress_queue.put(("Starting MCAD", None))

        if self.simulation_thread is not None:
            self.simulation_thread.start()
        else:
            logger.exception("Simulation thread is not initialized.")
            raise RuntimeError("Simulation thread is not initialized.")


    def _initialize_simulation_thread(self) -> None:
        self.simulation_thread = Thread(
            target=self.Simulation.run_simulation,
            daemon=False,
        )


    def _initialize_ui(self) -> None:
        self.ui = MyUserInterface(
            max_value=self.num_datapoints,
            title="MCAD Simulation",
            button_function=self._start_simulation,
            autostart=self.autostart
        )


    def _run_ui(self) -> None:

        if self.ui is not None and self.ui.root is not None:
            self.ui.root.mainloop()
        else:
            logger.exception("Failed to initialize the user interface.")
            raise RuntimeError("Failed to initialize the user interface.")

        if self.simulation_thread is not None:
            self.simulation_thread.join()

        if self.simulation_exception is not None:
            raise self.simulation_exception


    def _poll_queue(self) -> None:
        def _update_ui() -> None:
            if self.ui is not None:
                _value = self.simulated_points + self.dropped_points + self.existing_points + self.copied_points
                self.ui.update_progress(_value)
                self.ui.info_label.config(
                    text=(
                        f"Dropped: {self.dropped_points}\n"
                        f"Simulated: {self.simulated_points}\n"
                        f"Already  existing points: {self.existing_points}\n"
                        f"Copied points: {self.copied_points}"
                    )
                )

        if self.ui is not None:
            try:
                while True:
                    message_type, payload = self.progress_queue.get_nowait()

                    match message_type:
                        case "progress":
                            if payload:
                                self.simulated_points += 1
                            else:
                                self.dropped_points += 1
                            _update_ui()
                        case "point existing":
                            self.existing_points += 1
                            _update_ui()
                        case "point copied":
                            self.copied_points += 1
                            _update_ui()
                        case "DONE":
                            _update_ui()
                            self.ui.info_label.config(
                                text=(
                                    f"Dropped: {self.dropped_points}\n"
                                    f"Simulated: {self.simulated_points}\n"
                                    f"Already  existing points: {self.existing_points}\n"
                                    f"Copied points: {self.copied_points}\n"
                                    f"Finished"
                                )
                            )
                            self.ui.start_button_sim.config(text="Simulation Finished", state="disabled")
                            return
                        case "Starting MCAD":
                            self.ui.info_label.config(text="Starting MCAD...")
                            self.ui.start_button_sim.config(state="disabled", text="Running")
                        case "ERROR":
                            self.simulation_exception = payload

                            self.ui.info_label.config(
                                text=f"Simulation failed:\n{payload}"
                            )
                            self.ui.start_button_sim.config(
                                text="Simulation Failed",
                                state="disabled",
                            )
                            return

            except queue.Empty:
                pass

            self.ui.root.after(100, self._poll_queue)
        else:
            logger.error("User interface is not initialized.")
            raise RuntimeError("User interface is not initialized.")


    def create_csv_filepath(self) -> Path:
        current_time = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y_%H-%M-%S_%Z")
        filename = (
            f"Results_EMagnetic_simulation_"
            f"{self.num_base_designs}_"
            f"{self.num_lam_designs}_"
            f"{self.num_operation_points_per_dim}_"
            f"GenType-{self.gen_type}_"
            f"{current_time}.csv"
        )
        return PATH_DATA_MCAD_RESULTS / filename