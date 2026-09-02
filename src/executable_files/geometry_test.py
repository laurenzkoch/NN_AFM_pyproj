
from modules.process.motor_cad_processes.mcad_processing import test_boundary_conditions
from modules.generators.machine_generator import get_generator
from config.config_mcad_parameters import *


design_counter = 0
drop_counter = 0


for design in get_generator():

    design_counter += 1
    if design_counter % 2000 == 0:
        print(f"{design_counter} von {NUM_DATA_POINTS}; Testing {NUM_DATA_POINTS-design_counter} more")

    default_values = DEFAULT_VALUES.copy()
    for key, value in design.items():
        default_values[key] = value

    parameter_name: list[str] = list(default_values.keys())
    parameter_values: list[float] = list(default_values.values())

    if not test_boundary_conditions(parameter_name, parameter_values):
        print("________TEST FAILED FOR:____________")
        for _var, _value in default_values.items():
            print(f"{_var}: {_value}")
        drop_counter += 1

print(f"#############TEST COMPLETED: {drop_counter} from {design_counter} designs failed ##################")





