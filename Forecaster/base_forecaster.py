from abc import ABC, abstractmethod
import pandas as pd


class BaseForecaster(ABC):
    """
    Forecaster base class for anomaly score prediction.
    - fit: method to train the forecaster on the provided data.
    - predict: method to predict future anomaly scores on the provided data.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        pass
