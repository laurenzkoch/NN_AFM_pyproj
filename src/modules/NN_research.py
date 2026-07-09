from itertools import product
from typing import Generator

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from src.modules.ml_model_trainig import MLModelTrainer


class NNresearch(MLModelTrainer):

    def __init__(self) -> None:
        super().__init__()


    @staticmethod
    def create_model(layer_neurons: tuple) -> Pipeline:

        number_of_neurons_in_hl = layer_neurons

        mlp_classifier = MLPClassifier(
            hidden_layer_sizes=number_of_neurons_in_hl,
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42
        )

        model = Pipeline([
            ("scale", StandardScaler()),
            ("mlp", mlp_classifier)
        ])

        return model

    def model_variation(self) -> Generator[Pipeline, None, None]:

        neuron_variation = [(neurons,) for neurons in range(10, 100, 10)]

        layer_variations = [
            neuron_variation,
            [product(*neuron_variation, repeat=2)],
            [product(*neuron_variation, repeat=3)],
            [product(*neuron_variation, repeat=4)],
        ]

        for layers in layer_variations:
            for layer_neurons in layers:
                yield self.create_model(layer_neurons)