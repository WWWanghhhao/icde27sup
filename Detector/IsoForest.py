import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from Detector.base_detector import BaseDetector


class IsoForestDetector(BaseDetector):
    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto',
                 max_features=1.0, random_state=42, **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.max_features = max_features
        # self.scaler = RobustScaler()
        self.scaler = MinMaxScaler()
        self.score_scaler = MinMaxScaler()

        self.model = IsolationForest(n_estimators=self.n_estimators,
                                     max_samples=self.max_samples,
                                     contamination=self.contamination,
                                     max_features=self.max_features,
                                     random_state=self.random_state,
                                     n_jobs=-1)


    def fit(self, train_data):
        X_train_scaled = self.scaler.fit_transform(train_data)
        self.model.fit(X_train_scaled)
        train_scores_raw = -self.model.score_samples(X_train_scaled)
        train_scores_norm = self.score_scaler.fit_transform(train_scores_raw.reshape(-1, 1)).ravel()
        train_scores_series = pd.Series(train_scores_norm, index=train_data.index)
        return train_scores_series

    def predict(self, test_data):
        X_test_scaled = self.scaler.transform(test_data)
        test_scores_raw = -self.model.score_samples(X_test_scaled)
        test_scores_norm = self.score_scaler.transform(test_scores_raw.reshape(-1, 1)).ravel()
        test_scores_series = pd.Series(test_scores_norm, index=test_data.index)
        return test_scores_series
