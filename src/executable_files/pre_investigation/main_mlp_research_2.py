

from src.modules.NN_research import NNresearch
from src.config import config_paths
from src.modules.data_generating_pre_investigation import BasicDataGenerator
from datetime import datetime
import pickle

def main():

    output_dir = config_paths.PATH_DATA / "MLP_Research" / "Quantity of training data"
    output_dir.mkdir(parents=True, exist_ok=True)

    basic_data_generator = BasicDataGenerator()
    funcs = ["sin_exp" ,"sin", "exp", "linear", "quadratic"]
    neuron_layers = [
        (10,),
        (10, 10),
        (10, 10, 10),
        (100, 100),
        (100, 100, 100),
        (1, 1 ,1)
    ]

    for layer_neurons in neuron_layers:
        for func in funcs:
            data_points = [basic_data_generator.data_points(func=func, time_step=1 / i) for i in range(1, 400, 10)]
            xs = [data_points[i][["time"]] for i in range(len(data_points))]
            ys = [data_points[i][["y"]] for i in range(len(data_points))]


            researcher = NNresearch()
            model_data = researcher.quantity_of_datapoints(
                num_of_workers=5,
                xs=xs,
                ys=ys,
                func=func,
                layer_neurons=layer_neurons
            )


            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pkl_file = output_dir / f"model_data_{func}_{layer_neurons}_{timestamp}.pkl"

            with open(pkl_file, "wb") as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()