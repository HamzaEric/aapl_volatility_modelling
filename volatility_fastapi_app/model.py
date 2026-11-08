from __future__ import annotations
import os
from glob import glob
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from config import settings
from data import SQLRepository, wrangle_data
from yf_loader import get_daily_data


class GarchModel:
    def __init__(self, ticker: str, repo: SQLRepository, use_new_data: bool) -> None:
        self.ticker = ticker
        self.repo = repo
        self.use_new_data = use_new_data
        self.model_directory = settings.model_directory
        self.data: pd.Series | None = None
        self.model = None

    def wrangle_data(self, n_observations: int) -> None:
        if self.use_new_data:
            # Fetch new data and cache it
            df = get_daily_data(self.ticker)
            self.repo.insert_table(table_name=self.ticker, records=df, if_exists="replace")

        self.data = wrangle_data(repo=self.repo, ticker=self.ticker, n_observations=n_observations)

    def fit(self, p: int, q: int) -> None:
        model_setup = arch_model(self.data, p=p, q=q, rescale=False)
        self.model = model_setup.fit(disp="off")

    def __clean_prediction(self, horizon: int) -> dict[str, float]:
        forecast = self.model.forecast(horizon=horizon)
        variance = forecast.variance.iloc[-1]
        last_date = self.model.resid.index[-1]

        dates = pd.bdate_range(start=last_date + pd.DateOffset(days=1), periods=horizon)

        prediction = {}
        for i, date in enumerate(dates):
            prediction[date.isoformat()] = float(np.sqrt(variance.iloc[i]))
        return prediction

    def predict_volatility(self, horizon: int) -> dict[str, float]:
        return self.__clean_prediction(horizon=horizon)

    def dump(self) -> str:
        os.makedirs(self.model_directory, exist_ok=True)
        # Replace colons with dashes so Windows accepts the filename
        timestamp = datetime.now().isoformat().replace(":", "-")
        filepath = os.path.join(self.model_directory, f"{timestamp}_{self.ticker}.pkl")
        joblib.dump(self.model, filepath)
        return filepath

    def load(self) -> None:
        pattern = os.path.join(self.model_directory, f"*{self.ticker}.pkl")
        files = sorted(glob(pattern))
        if not files:
            raise Exception(f"No saved models found for ticker '{self.ticker}'.")

        latest_model_path = files[-1]
        self.model = joblib.load(latest_model_path)