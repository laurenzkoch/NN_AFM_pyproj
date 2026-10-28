from src.config.config_mcad_parameters import GEO_PARAMETERS
from src.modules.geometry.lambda_geometry import LambdaValues

MACHINE_DEFINITION: list[str] = [
    *GEO_PARAMETERS,
    *LambdaValues.columns(),
]
