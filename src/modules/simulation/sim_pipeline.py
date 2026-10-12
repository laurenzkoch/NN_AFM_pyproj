import logging
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config.config_mcad_parameters import MAP_SIZE, CSV_PARAMETERS
from src.config.config_paths import PATH_DATA_MCAD_RESULTS, PATH_MCAD_DATA, PATH_TEMP_FILES
from src.modules.Pipeline_and_GUI.events import PipelineEvent, EventType
from src.modules.Pipeline_and_GUI.pipeline import MyPipeline
from src.modules.data_managment.csv_io import create_csv_file
from src.modules.data_managment.file_io import delete_folder
from src.modules.process.eduvpn import EDUVPNProcessor
from src.modules.simulation.generators.machine_generator import AFMMachineGenerator
from src.modules.simulation.simulation import MCADSimulation

logger = logging.getLogger(__name__)


class SimulationPipeline(MyPipeline):

    def __init__(
            self,
            num_base_designs: int,
            num_lam_designs: int,
            start_eduvpn: bool = True,
            gen_type: str = "v1",
            csv_path: str | None = None,
    ) -> None:

        if num_base_designs < 1:
            raise ValueError("num_base_designs must be >= 1")

        if num_lam_designs < 1:
            raise ValueError("num_lam_designs must be >= 1")

        # eduVPN
        self.start_eduvpn = start_eduvpn
        self.eduvpn_processor: EDUVPNProcessor | None = None

        self.gen_type: str = gen_type
        self.num_base_designs: int = num_base_designs
        self.num_lam_designs: int = num_lam_designs

        super().__init__(
            service_name="MCAD Simulation",
            num_kernel=4,
        )

        self.status_text = "Ready"
        self.simulated_points: int = 0
        self.dropped_points: int = 0
        self.existing_points: int = 0
        self.copied_points: int = 0

        self.max_progress_steps: int = self._calculate_datapoints()
        self.num_workers = min(
            self.num_workers,
            self.max_progress_steps,
        )
        self.csv_filepath: Path = self.create_csv_filepath() if csv_path is None else PATH_DATA_MCAD_RESULTS / csv_path

        self.afmmachine_generator = AFMMachineGenerator(
            num_base_designs=num_base_designs,
            num_lam_designs=num_lam_designs,
            num_data_points=self.max_progress_steps,
        )
        self.Service = MCADSimulation(
            point_generator=self.afmmachine_generator.get_generator(_type=self.gen_type),
            batch_size=self.num_workers * math.prod(MAP_SIZE),
            progress_queue=self.progress_queue,
            log_queue=self.log_queue,
            csv_path=self.csv_filepath,
            num_workers=self.num_workers,
        )

    def start(self) -> None:

        try:
            if self.start_eduvpn:
                self.eduvpn_processor = EDUVPNProcessor().start()

            create_csv_file(
                path=self.csv_filepath,
                columns_list=CSV_PARAMETERS
            )
            self.run()
        finally:
            try:
                delete_folder(PATH_MCAD_DATA / PATH_TEMP_FILES)
            finally:
                if self.eduvpn_processor is not None:
                    self.eduvpn_processor.stop()


    def _calculate_datapoints(self) -> int:

        match self.gen_type:
            case "v1":
                return self.num_base_designs * self.num_lam_designs ** 3 * math.prod(MAP_SIZE)
            case "v2":
                return self.num_base_designs * self.num_lam_designs * math.prod(MAP_SIZE)
            case _:
                logger.exception("Invalid generator type.")
                raise ValueError("Invalid generator type.")

    def _queue_adapter(
            self,
            event: PipelineEvent,
    ) -> tuple[int, str, bool]:

        finished = False

        if event.event_type == EventType.STARTED:
            self.status_text = event.text

        elif event.event_type == EventType.PROGRESS:
            if event.outcome == "simulated":
                self.simulated_points += event.increment
            elif event.outcome == "dropped":
                self.dropped_points += event.increment
            elif event.outcome == "existing":
                self.existing_points += event.increment
            elif event.outcome == "copied":
                self.copied_points += event.increment

        elif event.event_type == EventType.FINISHED:
            self.status_text = event.text
            finished = True

        elif event.event_type == EventType.FAILED:
            self.status_text = event.text
            finished = True
        value = (
                self.simulated_points
                + self.dropped_points
                + self.existing_points
                + self.copied_points
        )
        return value, self.status_text, finished

    def create_csv_filepath(self) -> Path:
        current_time = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y_%H-%M-%S_%Z")
        filename = (
            f"Results_EMagnetic_simulation_"
            f"{self.num_base_designs}_"
            f"{self.num_lam_designs}_"
            f"{MAP_SIZE[0]}_"
            f"GenType-{self.gen_type}_"
            f"{current_time}.csv"
        )
        return PATH_DATA_MCAD_RESULTS / filename
