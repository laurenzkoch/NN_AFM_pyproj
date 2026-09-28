
from src.config.config_mcad_parameters import *

from src.modules.process.motor_cad_processes.mcad_processing import test_boundary_conditions
from src.modules.simulation.generators.machine_generator import AFMMachineGenerator


num_base_designs: int = 100
num_lam_designs: int = 100
num_operation_points_per_dim: int = 5
num_datapoints = num_base_designs * num_lam_designs**3 * num_operation_points_per_dim**3

i_max = 5
j_max = 5
_sum = 0

for i in range(3, i_max + 1):
    for j in range(3, j_max + 1):
        _sum += i * j**3 * 5**3
        with AFMMachineGenerator(
                num_base_designs=i,
                num_lam_designs=j,
                num_operation_points_per_dim=5,
                num_data_points=i * j**3 * 5**3
        ) as generator:

            design_counter = 0
            drop_counter = 0

            for design in generator.get_generator(_type="v1"):

                design_counter += 1

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

print(f"Datapoints: {_sum}")