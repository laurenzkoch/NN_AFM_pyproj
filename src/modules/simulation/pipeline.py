
from threading import Thread
from tkinter import TclError
from typing import Any
from queue import Queue
import queue
from datetime import datetime
from zoneinfo import ZoneInfo

from src.config.config_paths import *
from src.config.config_mcad_parameters import *

from src.modules.process.resources import determine_number_of_kernels
from src.modules.simulation.userinterface import MyUserInterface
from src.modules.generators.machine_generator import AFMMachineGenerator
from src.modules.simulation.run_simulation import run_simulation_pool
from src.modules.data_managment.file_io import delete_folder
from src.modules.process.eduvpn import EDUVPNProcessor
from src.modules.data_managment.csv_io import create_csv_file

class SimulationPipeline:

    def __init__(
            self,
            num_kernel: int,
            num_base_designs: int,
            num_sub_designs: int,
            num_subsub_designs: int,
            num_operation_points_per_dim: int,
            autostart: bool = False,
            start_eduvpn: bool = True
    ) -> None:
        self.autostart = autostart
        self.start_eduvpn = start_eduvpn
        self.ui: MyUserInterface | None = None
        self.simulation_thread: Thread | None = None
        self.simulation_exception: Any | None = None
        self.simulated_points: int = 0
        self.dropped_points: int = 0
        self.progress_queue: Queue[tuple[str, bool | None | Exception]] | None = None
        self.eduvpn_processor: EDUVPNProcessor | None  = None
        self.num_datapoints: int = num_base_designs * num_sub_designs * num_subsub_designs * num_operation_points_per_dim**3
        self.csv_filepath: Path = self.create_csv_filepath(
            num_base_designs,
            num_sub_designs,
            num_subsub_designs,
            num_operation_points_per_dim
        )
        self.num_base_designs = num_base_designs
        self.num_sub_designs = num_sub_designs
        self.num_subsub_designs = num_subsub_designs
        self.num_operation_points_per_dim = num_operation_points_per_dim
        self.num_workers = min(
            determine_number_of_kernels(num_kernel),
            self.num_datapoints
        )
        self.afmmachine_generator = AFMMachineGenerator(
            num_base_designs=num_base_designs,
            num_sub_designs=num_sub_designs,
            num_subsub_designs=num_subsub_designs,
            num_operation_points_per_dim=num_operation_points_per_dim
        )

    def start(self) -> None:
        if self.start_eduvpn:
            self.eduvpn_processor = EDUVPNProcessor().start()
        self.progress_queue = Queue()

        if self.csv_filepath is not None:
            create_csv_file(
                path=self.csv_filepath,
                columns_list=INPUT_NEURON_LIST + OP_PARAMETERS + OUTPUT_NEURON_LIST
            )
        else:
            raise RuntimeError("CSV file path is None.")

        self._initialize_simulation_thread()
        self._initialize_ui()
        self._initialize_progress_tracking()
        self._start_threads()


    def __enter__(self) -> "SimulationPipeline":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        delete_folder(PATH_MCAD_DATA / PATH_TEMP_FILES)
        if self.eduvpn_processor is not None:
            self.eduvpn_processor.__exit__(exc_type, exc_val, exc_tb)
        if self.ui is not None and self.ui.root is not None:
            try:
                self.ui.root.destroy()
            except TclError:
                pass

    def _initialize_progress_tracking(self) -> None:
        if self.progress_queue is not None:
            self._poll_queue()
        else:
            raise RuntimeError("Progress queue is not initialized.")


    def _start_simulation(self) -> None:
        if self.progress_queue is not None:
            self.progress_queue.put(("Starting MCAD", None))
        else:
            raise RuntimeError("Progress queue is not initialized.")

        if self.simulation_thread is not None:
            self.simulation_thread.start()
        else:
            raise RuntimeError("Simulation thread is not initialized.")


    def _initialize_simulation_thread(self) -> None:
        def _run_simulation() -> None:
            try:
                run_simulation_pool(
                    self.afmmachine_generator.get_generator(),
                    self.progress_queue,
                    self.csv_filepath,
                    self.num_workers,
                )
            except Exception as exc:
                if self.progress_queue is not None:
                    self.progress_queue.put(("ERROR", exc))
                else:
                    raise RuntimeError("Progress queue is not initialized.") from exc

        if self.progress_queue is not None:
            self.simulation_thread = Thread(
                target=_run_simulation,
                daemon=False,
            )
        else:
            raise RuntimeError("Progress queue is not initialized.")


    def _initialize_ui(self) -> None:
        self.ui = MyUserInterface(
            max_value=self.num_datapoints,
            title="MCAD Simulation",
            button_function=self._start_simulation,
            autostart=self.autostart
        )


    def _start_threads(self) -> None:

        if self.ui is not None and self.ui.root is not None:
            self.ui.root.mainloop()
        else:
            raise RuntimeError("Failed to initialize the user interface.")

        if self.simulation_thread is not None:
            self.simulation_thread.join()

        if self.simulation_exception is not None:
            raise self.simulation_exception


    def _poll_queue(self) -> None:
        if self.progress_queue is not None and self.ui is not None:
            try:
                while True:
                    message_type, payload = self.progress_queue.get_nowait()

                    match message_type:
                        case "progress":
                            if payload:
                                self.simulated_points += 1
                            else:
                                self.dropped_points += 1
                            value = self.simulated_points + self.dropped_points

                            self.ui.update_progress(value)
                            self.ui.info_label.config(
                                text=(
                                    f"Dropped: {self.dropped_points}\n"
                                    f"Simulated: {self.simulated_points}"
                                )
                            )
                        case "DONE":
                            self.ui.info_label.config(
                                text=(
                                    f"Dropped: {self.dropped_points}\n"
                                    f"Simulated: {self.simulated_points}\n"
                                    "Finished"
                                )
                            )
                            self.ui.start_button_sim.config(text="Simulation Finished", state="disabled")
                            if self.autostart:
                                self.ui.root.destroy()
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
                            if self.autostart:
                                self.ui.root.destroy()
                            return

            except queue.Empty:
                pass

            self.ui.root.after(100, self._poll_queue)
        else:
            raise RuntimeError("Progress queue or user interface is not initialized.")


    @staticmethod
    def create_csv_filepath(
        num_base_designs: int,
        num_sub_designs: int,
        num_subsub_designs: int,
        num_operation_points_per_dim: int
    ) -> Path:
        current_time = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y_%H-%M-%S_%Z")
        filename = (
            f"Results_EMagnetic_simulation_"
            f"{num_base_designs}_"
            f"{num_sub_designs}_"
            f"{num_subsub_designs}_"
            f"{num_operation_points_per_dim}_"
            f"{current_time}.csv"
        )
        return PATH_DATA_MCAD_RESULTS / filename