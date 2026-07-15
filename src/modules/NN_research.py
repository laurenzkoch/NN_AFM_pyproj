
from itertools import product
from pathlib import Path
from typing import Generator
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.pipeline import Pipeline

from src.modules.ml_model_trainig import MLModelTrainer, MLModel
from src.modules.plotting import Plotter
from src.modules.process_managing import ProcessManager


class NNresearch(MLModelTrainer):

    def __init__(self) -> None:
        super().__init__()


    def full_vectorial_training(
            self,
            num_of_workers: int = 4,
            max_num_of_layers: int = 3,
            min_neuron: int = 1,
            max_neuron: int = 201,
            neuron_step: int = 15,
            xs: list[pd.DataFrame] | None = None,
            ys: list[pd.DataFrame] | None = None,
            data_name: str = ""
    ) -> list[MLModel]:

        if xs is None or ys is None:
            raise ValueError("xs and ys must be provided")

        n_pairs = len(xs)
        n_var = len(range(min_neuron, max_neuron, neuron_step))
        total = n_pairs * sum(n_var ^ L for L in range(1, max_num_of_layers + 1))
        print(f"Total number of combinations: {total}")

        model_data: list[MLModel] = []

        process_manager = ProcessManager()
        num_of_workers = process_manager.set_number_of_workers(number_of_kernels=num_of_workers)

        neuron_variation = list(range(min_neuron, max_neuron, neuron_step))


        def task_list() -> Generator[tuple[pd.DataFrame, pd.DataFrame, tuple[int, ...], str], None, None]:

            if xs is None or ys is None:
                raise ValueError("xs and ys must be provided")

            for number_of_layers in range(1, max_num_of_layers + 1):
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
                model_data.append(future.result())

        return model_data


    def layer_neuron_variation(
            self,
            max_num_of_layers: int = 3,
            min_neuron: int = 1,
            max_neuron: int = 201,
            neuron_step: int = 15
    ) -> Generator[Pipeline, None, None]:

        neuron_variation = list(range(min_neuron, max_neuron, neuron_step))

        for number_of_layers in range(1, max_num_of_layers + 1):
            for layer_neurons in product(neuron_variation, repeat=number_of_layers):
                yield self.create_model(layer_neurons)

    @staticmethod
    def _func(data_name: str, x: pd.DataFrame, y: pd.DataFrame, layer_neurons: tuple = (100, 100, 100)) -> MLModel:

        trained_model = MLModelTrainer.train_model(
            data_name=data_name,
            x=x,
            y=y,
            test_split=0.2,
            model=MLModelTrainer.create_model(layer_neurons=layer_neurons)
        )

        return trained_model


    def neuron_variation_training(
            self,
            x: pd.DataFrame,
            y: pd.DataFrame,
            num_of_workers: int = 4,
            max_num_of_layers: int = 3,
            min_neuron: int = 1,
            max_neuron: int = 201,
            neuron_step: int = 15,
            func: str = "sin"
    ) -> list[MLModel]:

        model_data: list[MLModel] = []

        process_manager = ProcessManager()
        num_of_workers = process_manager.set_number_of_workers(number_of_kernels=num_of_workers)

        with ProcessPoolExecutor(max_workers=num_of_workers) as executor:
            futures = [
                executor.submit(
                    self._func,
                    x,
                    y,
                    model,
                    func
                ) for model in self.layer_neuron_variation(
                    max_num_of_layers=max_num_of_layers,
                    min_neuron=min_neuron,
                    max_neuron=max_neuron,
                    neuron_step=neuron_step
                )
            ]
            for future in futures:
                future.result()
                model_data.append(future.result())
        return model_data


    def quantity_of_datapoints(
            self,
            num_of_workers: int = 4,
            xs: list[pd.DataFrame] | None = None,
            ys: list[pd.DataFrame] | None = None,
            func: str = "sin",
            layer_neurons: tuple[int, ...] = (10, 10, 10),
    ) -> list[MLModel]:

        if xs is None or ys is None:
            raise ValueError("xs and ys must be provided")

        model_data: list[MLModel] = []

        process_manager = ProcessManager()
        num_of_workers = process_manager.set_number_of_workers(number_of_kernels=num_of_workers)

        with ProcessPoolExecutor(max_workers=num_of_workers) as executor:
            futures = [
                executor.submit(
                    self._func,
                    x,
                    y,
                    self.create_model(layer_neurons=layer_neurons),
                    func
                ) for (x, y) in zip(xs, ys)
            ]
            for future in futures:
                future.result()
                model_data.append(future.result())
        return model_data





    @staticmethod
    def create_plot_of_model(model: MLModel, x: pd.DataFrame, y: pd.DataFrame, idx: int, func: str, variation_type: str) -> None:

        plotter = Plotter()

        match func:
            case "sin_exp":
                function = "100 * sin(0.8 * t) * exp(-t/50)"
            case "sin":
                function = "100 * sin(0.8 * t)"
            case "exp":
                function = "100 * exp(-t/50)"
            case "linear":
                function = "1/50 * t + 10"
            case "quadratic":
                function = "1/50 * t^2 + 10"
            case _:
                function = ""

        fig, ax = plotter.create_fig_matplotlib(
            xlabel="time [s]",
            ylabel="y",
            title=f"{variation_type}"
        )
        ax = plotter.add_scatter_to_fig(ax=ax, x_column=model.x_test, y_column=model.y_pred, colors="r", label="y_pred")
        ax = plotter.add_plot_to_fig(ax=ax, x_column=x, y_column=y, colors="b", label="real data")
        ax = plotter.add_text_to_fig(
            ax=ax,
            text=f"R2 Score: {model.R2_Score:.4f}\n"
                 f"Mean Squared Error: {model.Mean_Squared_Error:.4f}\n"
                 f"Score: {model.Score:.4f}\n"
                 f"Layer Sizes: {model.layer_sizes}\n"
                 f"Split: 0.20\n"
                 f"Quantity of training points: {int(len(x) * 0.8)}\n"
                 f"Function: {function}\n"
                 f"Computing Time: {model.computing_time:.4f} s",

        )

        plotter.add_legend_to_fig(ax=ax)

        plotter.save_plot(
            fig=fig,
            path_plots=Path(f"MLP_{variation_type}") / f"MLP_{func}_{model.layer_sizes}",
            filename=f"{func}_{idx}.png"
        )

