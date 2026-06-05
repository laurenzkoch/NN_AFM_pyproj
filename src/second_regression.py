"""- RandomForestRegressor
- GradientBoostingRegressor
- HistGradientBoostingRegressor
- GaussianProcessRegressor
- MLPRegressor"""

import utils

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
from matplotlib import pyplot as plt
from pandas import DataFrame
from typing import Any, cast

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor


class SECONDREGRESSION:
    def __init__(self):

        rfr = RandomForestRegressor()
        gbr = GradientBoostingRegressor()
        hgbr = HistGradientBoostingRegressor()
        gpr = GaussianProcessRegressor()
        nnr = MLPRegressor()


        self.models = [rfr, gbr, hgbr, gpr, nnr]

