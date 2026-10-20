from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    STARTED = "started"
    PROGRESS = "progress"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"

    SIMULATION_STARTED = "simulation_started"
    SIMULATION_FINISHED = "simulation_finished"


@dataclass(frozen=True)
class PipelineEvent:
    event_type: EventType
    text: str = ""
    increment: int = 0
    outcome: str | None = None

    max_steps: int | None = None
    combination_index: int | None = None
