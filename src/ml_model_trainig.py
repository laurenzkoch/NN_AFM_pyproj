import pandas as pd
from typing import Any, cast
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class MLModelTrainer:

    def __init__(self) -> None:
        pass


    @staticmethod
    def train_model(
            x: pd.DataFrame,
            y: pd.DataFrame,
            test_split: float,
            model: Any
    ) -> tuple[Any, pd.DataFrame, pd.DataFrame]:

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=42)
        model.fit(x_train, y_train)

        return model, x_test, y_test

    @staticmethod
    def evaluate_model(
            model: Any,
            x_test: pd.DataFrame,
            y_test: pd.DataFrame
    ) -> pd.DataFrame:

        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        mean_squared_error_value = cast(float, mean_squared_error(y_test, y_pred))

        trained_and_evaluated_model = pd.DataFrame([
            {
                "Model": model.__class__.__name__,
                "R^2 Score": r2,
                "Mean Squared Error": mean_squared_error_value,
                "y_pred": y_pred,
                "x_test": x_test
             }
        ])

        return trained_and_evaluated_model