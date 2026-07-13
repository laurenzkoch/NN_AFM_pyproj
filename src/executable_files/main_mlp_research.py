

from src.modules.NN_research import NNresearch
from src.config import config_paths
from src.modules.data_generating import BasicDataGenerator
from datetime import datetime
import pickle

def main():

    output_dir = config_paths.PATH_DATA / "MLP_Research"
    output_dir.mkdir(parents=True, exist_ok=True)

    funcs = ["sin", "exp", "linear", "quadratic"]
    basic_data_generator = BasicDataGenerator()

    for func in funcs:
        data_points = basic_data_generator.data_points(func=func)
        x, y = data_points[["time"]], data_points[["y"]]

        researcher = NNresearch()
        model_data = researcher.training(
            x=x,
            y=y,
            max_num_of_layers=5,
            num_of_workers=5,
            min_neuron=1,
            max_neuron=350,
            neuron_step=5,
            func=func
        )


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pkl_file = output_dir / f"model_data_{func}_{timestamp}.pkl"

        with open(pkl_file, "wb") as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()