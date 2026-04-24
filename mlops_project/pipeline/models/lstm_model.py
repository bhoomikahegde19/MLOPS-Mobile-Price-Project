from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import RANDOM_STATE
from ..data.preprocess import build_preprocessor, transform_with_preprocessor
from ..training.dataset import TorchTabularDataset


class TabularLSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


@dataclass
class LSTMPriceModel:
    preprocessor: object | None = None
    model: TabularLSTMRegressor | None = None
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 5e-4

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMPriceModel":
        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        self.preprocessor = build_preprocessor(X)
        X_processed = self.preprocessor.fit_transform(X)
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
        X_processed = np.asarray(X_processed, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32)

        dataset = TorchTabularDataset(X_processed, y_array)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = TabularLSTMRegressor(input_size=1, hidden_size=48)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for _ in range(self.epochs):
            for batch_features, batch_targets in loader:
                optimizer.zero_grad()
                predictions = self.model(batch_features.unsqueeze(-1))
                loss = criterion(predictions, batch_targets.unsqueeze(-1))
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X: pd.DataFrame):
        if self.preprocessor is None or self.model is None:
            raise ValueError("LSTM model has not been trained.")
        transformed = transform_with_preprocessor(self.preprocessor, X)
        tensor = torch.tensor(np.asarray(transformed, dtype=np.float32)).unsqueeze(-1)
        self.model.eval()
        with torch.no_grad():
            return self.model(tensor).squeeze(-1).cpu().numpy()
