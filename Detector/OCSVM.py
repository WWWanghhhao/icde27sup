import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from Detector.base_detector import BaseDetector


class OCSVMDetector(BaseDetector):
    def __init__(self, kernel='rbf', nu=0.1, gamma='scale', **kwargs):
        """
        Args:
            kernel (str): OCSVM 的核函数 (e.g., 'rbf', 'linear').
            nu (float): 异常比例的上限 (0 到 1 之间). 这是最重要的参数。
            gamma (str or float): 'rbf' 核的系数。
        """
        super().__init__(**kwargs)
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.scaler = RobustScaler()
        self.score_scaler = MinMaxScaler()
        self.model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)

    def fit(self, data: pd.DataFrame) -> pd.Series:
        X_scaled = self.scaler.fit_transform(data)
        self.model.fit(X_scaled)
        train_scores_raw = -self.model.score_samples(X_scaled)
        train_scores_norm = self.score_scaler.fit_transform(train_scores_raw.reshape(-1, 1)).ravel()
        return pd.Series(train_scores_norm, index=data.index)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        X_scaled = self.scaler.transform(data)
        test_scores_raw = -self.model.score_samples(X_scaled)
        test_scores_norm = self.score_scaler.transform(test_scores_raw.reshape(-1, 1)).ravel()
        return pd.Series(test_scores_norm, index=data.index)
