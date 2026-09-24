"""Categorical and numerical encoder helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FeatureEncoderRegistry:
    """Registry managing fitted label encoders and scalers for .pkl serialization."""

    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Fit and transform specified categorical columns."""
        res = df.copy()
        for col in columns:
            if col in res.columns:
                le = LabelEncoder()
                res[col] = le.fit_transform(res[col].astype(str).fillna("Unknown"))
                self.encoders[col] = le
        return res

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe using previously fitted encoders."""
        res = df.copy()
        for col, le in self.encoders.items():
            if col in res.columns:
                known_classes = set(le.classes_)
                res[col] = res[col].astype(str).map(lambda s: s if s in known_classes else "Unknown")
                if "Unknown" not in known_classes:
                    res[col] = 0
                else:
                    res[col] = le.transform(res[col])
        return res

    def save_pkl(self, filepath: Path) -> None:
        """Serialize encoder state to .pkl file."""
        with open(filepath, "wb") as f:
            pickle.dump(self.encoders, f)

    def load_pkl(self, filepath: Path) -> None:
        """Load encoder state from .pkl file."""
        with open(filepath, "rb") as f:
            self.encoders = pickle.load(f)
