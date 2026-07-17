
from itertools import product
import pickle
from typing import Generator
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import root_scalar

from src.modules.ml_model_trainig import MLModelTrainer, MLModel
from src.modules.process_managing import ProcessManager
from src.config import config_paths


class NNresearch(MLModelTrainer):

    def __init__(self) -> None:
        super().__init__()

    def full_vectorial_training(
            self,
            num_of_workers: int = 4,
            min_num_of_layers: int = 1,
            max_num_of_layers: int = 3,
            min_neuron: int = 1,
            max_neuron: int = 201,
            number_neuron_variations: int = 1000,
            xs: list[pd.DataFrame] | None = None,
            ys: list[pd.DataFrame] | None = None,
            data_name: str = ""
    ) -> None:

        if xs is None or ys is None:
            raise ValueError("xs and ys must be provided")

        def find_lenght_of_neuron_variation() -> int:
            if max_num_of_layers + 1 <= min_num_of_layers:
                raise ValueError("b muss größer als a sein.")

            if abs(number_neuron_variations - (max_num_of_layers + 1 - min_num_of_layers)) < 1e-12:
                return 1

            def equation(x):
                return sum(x ** i for i in range(min_num_of_layers, max_num_of_layers + 1)) - number_neuron_variations

            f_lower = equation(0)
            f_upper = equation(1000)

            if f_lower * f_upper > 0:
                raise ValueError(
                    "Im angegebenen Intervall wurde keine eindeutige Nullstelle "
                    "eingeschlossen. Passe lower und upper an."
                )

            result = root_scalar(
                equation,
                bracket=(0, 1000),
                method="brentq",
                xtol=1e-12
            )

            return round(result.root)

        n_var_calc = find_lenght_of_neuron_variation()
        neuron_variation = list(range(min_neuron, max_neuron, (max_neuron-min_neuron) // n_var_calc))
        n_var = len(neuron_variation)
        print(f"Number of neuron variations: {n_var}; calculated: {n_var_calc}")

        n_pairs = min(len(xs), len(ys))
        number_neuron_variations_calc = sum(
            n_var ** number_of_layers
            for number_of_layers in range(min_num_of_layers, max_num_of_layers + 1)
        )
        total = n_pairs * number_neuron_variations_calc
        print(f"Total number of combinations: {total}")

        process_manager = ProcessManager()
        num_of_workers = process_manager.set_number_of_workers(number_of_kernels=num_of_workers)


        def task_list() -> Generator[tuple[pd.DataFrame, pd.DataFrame, tuple[int, ...], str], None, None]:
            if xs is None or ys is None:
                raise ValueError("xs and ys must be provided")

            for number_of_layers in range(min_num_of_layers, max_num_of_layers + 1):
                for layer_neurons in product(neuron_variation, repeat=number_of_layers):
                    for (x, y) in zip(xs, ys):
                        yield x, y, layer_neurons, data_name

        with ProcessPoolExecutor(max_workers=num_of_workers) as executor:
            futures = [
                executor.submit(
                    self._func,
                    data_name,
                    x,
                    y,
                    layer_neurons
                ) for (x, y, layer_neurons, data_name) in task_list()
            ]
            for future in futures:
                future.result()


    @staticmethod
    def _func(data_name: str, x: pd.DataFrame, y: pd.DataFrame, layer_neurons: tuple = (100, 100, 100)) -> None:

        print(f"func call: {layer_neurons} and {len(x)} data points")
        trained_model = MLModelTrainer.train_model(
            data_name=data_name,
            x=x,
            y=y,
            test_split=0.2,
            model=MLModelTrainer.create_model(layer_neurons=layer_neurons)
        )
        trained_model.Model = None

        output_dir = config_paths.PATH_DATA / "MLP_Research" / "Pickel_files"
        output_dir.mkdir(parents=True, exist_ok=True)

        if trained_model.x_train is None:
            raise ValueError("x_train must be provided")

        pkl_file = output_dir / f"model_data_{data_name}_{trained_model.layer_sizes}_{len(trained_model.x_train)}.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump(trained_model, f, protocol=pickle.HIGHEST_PROTOCOL)






    # @staticmethod
    # def create_plot_of_model(model: MLModel, x: pd.DataFrame, y: pd.DataFrame, idx: int, func: str, variation_type: str) -> None:
    #
    #     plotter = Plotter()
    #
    #     match func:
    #         case "sin_exp":
    #             function = "100 * sin(0.8 * t) * exp(-t/50)"
    #         case "sin":
    #             function = "100 * sin(0.8 * t)"
    #         case "exp":
    #             function = "100 * exp(-t/50)"
    #         case "linear":
    #             function = "1/50 * t + 10"
    #         case "quadratic":
    #             function = "1/50 * t^2 + 10"
    #         case _:
    #             function = ""
    #
    #     fig, ax = plotter.create_fig_matplotlib(
    #         xlabel="time [s]",
    #         ylabel="y",
    #         title=f"{variation_type}"
    #     )
    #     ax = plotter.add_scatter_to_fig(ax=ax, x_column=model.x_test, y_column=model.y_pred, colors="r", label="y_pred")
    #     ax = plotter.add_plot_to_fig(ax=ax, x_column=x, y_column=y, colors="b", label="real data")
    #     ax = plotter.add_text_to_fig(
    #         ax=ax,
    #         text=f"R2 Score: {model.R2_Score:.4f}\n"
    #              f"Mean Squared Error: {model.Mean_Squared_Error:.4f}\n"
    #              f"Score: {model.Score:.4f}\n"
    #              f"Layer Sizes: {model.layer_sizes}\n"
    #              f"Split: 0.20\n"
    #              f"Quantity of training points: {int(len(x) * 0.8)}\n"
    #              f"Function: {function}\n"
    #              f"Computing Time: {model.computing_time:.4f} s",
    #
    #     )
    #
    #     plotter.add_legend_to_fig(ax=ax)
    #
    #     plotter.save_plot(
    #         fig=fig,
    #         path_plots=Path(f"MLP_{variation_type}") / f"MLP_{func}_{model.layer_sizes}",
    #         filename=f"{func}_{idx}.png"
    #     )

