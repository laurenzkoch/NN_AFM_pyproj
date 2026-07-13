from itertools import product
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

    def model_variation(
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
    def _func(idx: int, x: pd.DataFrame, y: pd.DataFrame, model: Pipeline, func: str) -> MLModel:
        trained_model, x_test, y_test = MLModelTrainer.train_model(x=x, y=y, test_split=0.2, model=model)
        trained_a_evaluated_model = MLModelTrainer.evaluate_model(trained_model=trained_model, x_test=x_test, y_test=y_test)
        if trained_a_evaluated_model.R2_Score > -0.01:
            NNresearch.create_plot_of_model(model=trained_a_evaluated_model, x=x, y=y, idx=idx, func=func)
        return trained_a_evaluated_model


    def training(
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
                    idx,
                    x,
                    y,
                    model,
                    func
                ) for idx, model in enumerate(
                    self.model_variation(
                        max_num_of_layers=max_num_of_layers,
                        min_neuron=min_neuron,
                        max_neuron=max_neuron,
                        neuron_step=neuron_step
                    ),
                    start=1
                )
            ]
            for future in futures:
                future.result()
                model_data.append(future.result())
        return model_data


    @staticmethod
    def create_plot_of_model(model: MLModel, x: pd.DataFrame, y: pd.DataFrame, idx: int, func: str) -> None:

        plotter = Plotter()

        fig, ax = plotter.create_fig_matplotlib(
            xlabel="time [s]",
            ylabel="y",
            title=f"Layer Sizes: {model.layer_sizes}"
        )
        ax = plotter.add_scatter_to_fig(ax=ax, x_column=model.x_test, y_column=model.y_pred, colors="r", label="y_pred")
        ax = plotter.add_plot_to_fig(ax=ax, x_column=x, y_column=y, colors="b", label="real data")
        ax = plotter.add_text_to_fig(ax=ax, text=f"R2 Score: {model.R2_Score:.4f}\n"
                                                 f"Mean Squared Error: {model.Mean_Squared_Error:.4f}\n"
                                                 f"Score: {model.Score:.4f}\nLayer Sizes: {model.layer_sizes}")

        plotter.add_legend_to_fig(ax=ax)

        plotter.save_plot(
            fig=fig,
            path_plots=f"MLP_{func}",
            filename=f"{func}_{idx}.png"
        )
