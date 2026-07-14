from dataclasses import dataclass

import pandas as pd
from typing import Any, cast
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

@dataclass
class MLModel:
    Model_name: str
    R2_Score: float
    Mean_Squared_Error: float
    Score: float
    layer_sizes: tuple[int, ...]
    y_pred: pd.Series | pd.DataFrame
    x_test: pd.Series | pd.DataFrame
    computing_time: float = 0.0

class MLModelTrainer:

    @staticmethod
    def create_model(layer_neurons: tuple = (100, 100, 100)) -> Pipeline:

        mlp_classifier = MLPRegressor(
            hidden_layer_sizes=layer_neurons,
            activation="relu",
            solver="adam",
            alpha=0,
            learning_rate_init=1e-3,
            max_iter=5000,
            early_stopping=True,
            n_iter_no_change=50,
            random_state=42
        )

        model = Pipeline([
            ("scale", StandardScaler()),
            ("mlp", mlp_classifier)
        ])

        return model

    @staticmethod
    def train_model(
            x: pd.DataFrame,
            y: pd.DataFrame,
            test_split: float,
            model: Any
    ) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
        
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y.iloc[:, 0],
            test_size=test_split,
            random_state=42,
            shuffle=True
        )
        trained_model = model.fit(x_train, y_train)
        del model, x_train, y_train

        return trained_model, x_test, y_test

    @staticmethod
    def evaluate_model(
            trained_model: Any,
            x_test: pd.DataFrame,
            y_test: pd.DataFrame
    ) -> MLModel:

        y_pred = trained_model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        mean_squared_error_value = cast(float, mean_squared_error(y_test, y_pred))
        score = trained_model.score(x_test, y_test)
        layer_sizes = trained_model.named_steps["mlp"].hidden_layer_sizes

        trained_and_evaluated_model = MLModel(
            Model_name=trained_model.__class__.__name__,
            R2_Score=r2,
            Mean_Squared_Error=mean_squared_error_value,
            Score=score,
            layer_sizes=layer_sizes,
            y_pred=y_pred,
            x_test=x_test
        )

        del trained_model

        return trained_and_evaluated_model