import data_generating
import pandas as pd

#list with all output parameters (str)
output_parameters = ["ShaftTorque"]

#data sets for the input parameters
input_parameters = [
    {"name": "PeakCurrent", "min_value": 1, "max_value": 100, "step_size": 50},
    {"name": "PhaseAdvance", "min_value": 0, "max_value": 90, "step_size": 45}
]



with data_generating.MCADDataGenerator() as data_generator:
    data_generator.open_motorcad_file("base_first_data.mot")
    value_df = data_generator.take_input_parameter(
        pd.DataFrame(
            input_parameters,
            columns=["name", "min_value", "max_value", "step_size"]
        )
    )
    results_df = data_generator.execute_mcad_simulation(
        value_df,
        output_parameters=output_parameters
    )
    data_generator.export_results(results_df)
