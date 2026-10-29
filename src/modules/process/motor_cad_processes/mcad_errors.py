"""Application-level exceptions for Motor-CAD execution."""


class MCADExecutionError(RuntimeError):
    """Base exception for failures during Motor-CAD execution."""


class MCADCommunicationError(MCADExecutionError):
    """Communication with the Motor-CAD instance failed."""


class MCADCalculationError(MCADExecutionError):
    """Motor-CAD reported a calculation failure."""


class MCADInitializationError(MCADExecutionError):
    """A Motor-CAD instance could not be initialized."""


class MCADResultError(MCADExecutionError):
    """A calculation produced missing or invalid results."""


class MCADCleanupError(MCADExecutionError):
    """A Motor-CAD instance could not be closed correctly."""
