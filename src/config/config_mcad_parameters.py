D_CURRENT_DENSITY: tuple[float, float] = (-10, 0)
Q_CURRENT_DENSITY: tuple[float, float] = (1, 25)
# shaft_speed_norm = shaft_speed_rpm / base_speed_rpm
# base_speed_rpm: no-load voltage-limit speed for ideal linear SVPWM
SHAFT_SPEED_NORM: tuple[float, float] = (0.05, 1.0)
REFERENCE_SPEED_RPM: float = 1000.0
MAP_SIZE: tuple[int, int, int] = (5, 5, 5)

POLE_NUMBER: int = 12
SLOT_NUMBER: int = 18
AIRGAP: tuple[float, float] = (0.75, 3)
STATOR_BORE: tuple[float, float] = (11, 100)

GEO_PARAMETERS_MIN_MAX_DICT = {
    "Airgap": AIRGAP,
    "Stator_Bore": STATOR_BORE,
}

LAMBDA_BOUNDS: dict[str, tuple[float, float]] = {
    # radial:       lambda_1 = D_Si / D_Sa
    "lambda_1": (0.2, 0.8),
    # tangential:   lambda_2 = beta_PM / beta_p = 2 * Magnet_Arc_[ED] / 360
    "lambda_2": (0.5, 0.9),
    # axial:        lambda_3 = g / l_PM
    "lambda_3": (0.2, 1.0),
    # tangential:   lambda_4 = b_S / tau_s = s*b_S / pi*D_Si
    "lambda_4": (0.3, 0.7),
    # axial:        lambda_5 = b_S / d_S
    "lambda_5": (0.1, 1.0),
    # radial:       lambda_6 = h_PM / (D_Sa - D_Si)/2
    "lambda_6": (0.6, 1.0),
    # radial:       lambda_7 = d_emb / ((D_Sa - D_Si)/2 - h_PM)
    "lambda_7": (0.0, 0.95),
    # tangential:   lambda_8 = Dicke des magnetischen Rückflusses / tau_p
    # = 2*p * (l_S - d_s) / (pi * D_Si * (1 + 1/lambda_1))
    "lambda_8": (0.1, 0.4),
    # axial:        lambda_9 = l_PM / l_R
    "lambda_9": (0.2, 1.0),
    "lambda_10": (0.1, 1.0),  # lambda_10 = b_tto / b_S
    "lambda_11": (0.0, 0.3),  # lambda_11 = d_ttd / d_S
    "lambda_12": (0.0, 1.0),  # lambda_12 = r_s / (b_S/2)
}

SIM_IN_GEO_PARAMETERS = [
    "Pole_Number",
    "Slot_Number",
    "Airgap",
    "Stator_Bore",
    "Stator_Lam_Dia",
    "AFM_D_Rotor",
    "Magnet_Arc_[ED]",
    "Magnet_Length",
    "Magnet_Thickness",
    "CircumferentialSegments",
    "Slot_Depth",
    "Slot_Width",
    "Slot_Opening",
    "Slot_Corner_Radius",
    "Tooth_Tip_Depth",
    "Magnet_Embed_Depth",
    "Rotor_Lam_Length",
    "Stator_Lam_Length",
    "Motor_Length",
    "MagPhases",
    "MagTurnsConductor",
    "MagThrow",
    "ParallelPaths",
    "WindingLayers",
    "MagWindingOffset",
    "NumberOfCoils",
]

OP_PARAMETERS = [
    "ShaftSpeed",
    "DCBusVoltage",
    "RMSPhaseCurrent_D",
    "RMSPhaseCurrent_Q",
    "RMSCurrentDensity",
    "PeakCurrent",
    "PhaseAdvance",
]

OUTPUT_NEURON_LIST = [
    "ShaftTorque",
    "SystemEfficiency",
    "BMean_AirGap",
    "ConductorLoss",
    "MagnetLoss",
    "StatorIronLoss_Total",
    "RotorIronLoss_Total",
]

CSV_DECIMAL_PLACES: int = 12
MCAD_RESULT_PARAMETERS: list[str] = (
    SIM_IN_GEO_PARAMETERS + OP_PARAMETERS + OUTPUT_NEURON_LIST
)

REFERENCE_CSV_PARAMETERS: list[str] = [
    "ReferenceSpeedRPM",
    "ReferenceKeFundamental",
    "ReferencePeakBackEMFPhase",
    "BaseSpeedRPM",
    "ShaftSpeedNorm",
]

CSV_PARAMETERS: list[str] = MCAD_RESULT_PARAMETERS + REFERENCE_CSV_PARAMETERS

GEO_PARAMETERS = [
    "Airgap",
    "Stator_Bore",
]

AXIS_LIST = [
    "Shaft_Speed_norm",
    "RMSPhaseCurrent_Q_density",
    "RMSPhaseCurrent_D_density",
]

DEFAULT_VALUES = {
    "AFM_D_Rotor": 150,
    "Stator_Lam_Dia": 150,
    "Stator_Bore": 70,
    "Airgap": 1,
    "Pole_Number": 12,
    "Slot_Number": 18,
    "Slot_Width": 7,
    "Slot_Opening": 3,
    "Slot_Depth": 18,
    "Slot_Corner_Radius": 0,
    "Tooth_Tip_Depth": 1,
    "Tooth_Tip_Angle": 30,
    "Motor_Length": 220,
    "Rotor_Lam_Length": 20,
    "Stator_Lam_Length": 50,
    "Magnet_Length": 5,
    "Magnet_Embed_Depth": 5,
    "Magnet_Thickness": 40,
    "CircumferentialSegments": 1,
    "Magnet_Arc_[ED]": 140,
    "Banding_Thickness": 0,
    "MagPhases": 3,
    "MagTurnsConductor": 50,
    "MagThrow": 1,
    "ParallelPaths": 1,
    "WindingLayers": 2,
    "MagWindingOffset": 1,
    "NumberOfCoils": 6,
    "Shaft_Dia": 10,
    "Shaft_Hole_Diameter": 0,
    "Shaft_Extension_[F]": 0,
    "Shaft_Extension_[R]": 0,
    "RMSCurrentDensity": 7.575,
    "PeakCurrent": 5,
    "ShaftSpeed": 3000,
    "PhaseAdvance": 0,
    "RMSPhaseCurrent_D": 0,
    "RMSPhaseCurrent_Q": 5.0,
    "DCBusVoltage": 800,
}
