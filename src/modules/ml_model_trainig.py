from dataclasses import dataclass, field
import time
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
    Model: Any | None = None
    layer_sizes: tuple[int, ...] = ()
    data_name: str = ""
    x: pd.Series | pd.DataFrame = field(default_factory=pd.DataFrame())
    y: pd.Series | pd.DataFrame = field(default_factory=pd.DataFrame())
    x_train: pd.Series | pd.DataFrame = field(default_factory=pd.DataFrame())
    y_train: pd.Series | pd.DataFrame = field(default_factory=pd.DataFrame())
    x_test: pd.Series | pd.DataFrame = field(default_factory=pd.DataFrame())
    y_test: pd.Series | pd.DataFrame = field(default_factory=pd.DataFrame())
    y_pred: pd.Series | pd.DataFrame = field(default_factory=pd.DataFrame())
    split: float = 0.2
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.0
    learning_rate_init: float = 1e-3
    max_iter: int = 5000
    early_stopping: bool = True
    n_iter_no_change: int = 50
    random_state: int = 42
    R2_pred: float | None = None
    R2_test_pred: float | None = None
    R2_train_pred: float | None = None
    mse: float | None = None
    mse_train: float | None = None
    mse_test: float | None = None
    computing_time: float = 0.0

class MLModelTrainer:

    @staticmethod
    def create_model(layer_neurons: tuple = (100, 100, 100)) -> Pipeline:

        mlp_regressor = MLPRegressor(
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
            ("mlp", mlp_regressor)
        ])

        return model

    @staticmethod
    def train_model(
            data_name: str,
            x: pd.DataFrame,
            y: pd.DataFrame,
            test_split: float,
            model: Any
    ) -> MLModel:

        start = time.perf_counter()

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y.iloc[:, :],
            test_size=test_split,
            random_state=42,
            shuffle=True
        )

        model = model.fit(x_train, y_train)

        y_test_pred = pd.DataFrame(model.predict(x_test), columns=y.columns, index=x_test.index)
        y_train_pred = pd.DataFrame(model.predict(x_train), columns=y.columns, index=x_train.index)
        y_pred = pd.DataFrame(model.predict(x), columns=y.columns, index=x.index)

        r2_test_pred = r2_score(y_test, y_test_pred)
        r2_train_pred = r2_score(y_train, y_train_pred)
        r2_pred = r2_score(y, y_pred)

        mse_test = cast(float, mean_squared_error(y_test, y_test_pred))
        mse_train = cast(float, mean_squared_error(y_train, y_train_pred))
        mse = cast(float, mean_squared_error(y, y_pred))

        trained_model = MLModel(
            Model_name=model.__class__.__name__,
            Model=model,
            layer_sizes=model.named_steps["mlp"].hidden_layer_sizes,
            data_name=data_name,
            x=x,
            y=y,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            y_pred=y_pred,
            R2_pred=r2_pred,
            R2_test_pred=r2_test_pred,
            R2_train_pred=r2_train_pred,
            mse=mse,
            mse_train=mse_train,
            mse_test=mse_test,
            computing_time=time.perf_counter() - start
        )

        return trained_model