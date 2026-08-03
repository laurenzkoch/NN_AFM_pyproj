

from modules.generators.NN_research import NNresearch
from src.config import config_paths
from modules.generators.data_generating_pre_investigation import BasicDataGenerator
from datetime import datetime
import pickle

def main():

    output_dir = config_paths.PATH_DATA / "MLP_Research" / "Neuron Variation"
    output_dir.mkdir(parents=True, exist_ok=True)

    funcs = ["sin_exp", "sin", "exp", "linear", "quadratic"]
    basic_data_generator = BasicDataGenerator()

    for func in funcs:
        data_points = basic_data_generator.data_points(func=func)
        x, y = data_points[["time"]], data_points[["y"]]

        researcher = NNresearch()
        model_data = researcher.neuron_variation_training(x=x, y=y, num_of_workers=5, max_num_of_layers=2, min_neuron=1,
                                                          max_neuron=10, neuron_step=2, func=func)


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pkl_file = output_dir / f"model_data_{func}_{timestamp}.pkl"

        with open(pkl_file, "wb") as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()