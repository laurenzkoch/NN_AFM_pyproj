import pandas as pd
import plotly.express as px


def plot_datapoints_plotly(
        x_column: pd.Series,
        y_column: pd.Series,
        z_column: pd.Series | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
) -> None:
    if z_column is None:
        fig = px.scatter(
            x=x_column,
            y=y_column,
            title=title,
            labels={
                "x": xlabel,
                "y": ylabel,
            }
        )
        fig.update_layout(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
        )
    else:
        fig = px.scatter_3d(
            x=x_column,
            y=y_column,
            z=z_column,
            color=z_column,
            title=title,
            labels={
                "x": xlabel,
                "y": ylabel,
                "z": zlabel,
                "color": zlabel or z_column.name,
            }
        )
        fig.update_layout(
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                zaxis_title=zlabel,
            )
        )

    fig.show(renderer="plotly_mimetype")
