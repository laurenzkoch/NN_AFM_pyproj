from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    STARTED = "started"
    PROGRESS = "progress"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(frozen=True)
class PipelineEvent:
    event_type: EventType
    text: str = ""
    increment: int = 0
    outcome: str | None = None
