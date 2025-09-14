"""Utility helpers for the Pyro-meets-SBI tutorial notebooks.

This module provides convenient loaders for the cookie dataset that return both
pandas DataFrames and PyTorch tensors ready for modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from generate_data import (
    DEFAULT_LOCATION_RATES,
    get_default_data_path,
    load_or_generate_cookie_data,
)


@dataclass(frozen=True)
class CookieDataset:
    cookies: pd.DataFrame
    location_stats: pd.DataFrame
    locations: torch.Tensor
    chips: torch.Tensor
    n_locations: int
    location_rates: np.ndarray
    sample_sizes: np.ndarray
    true_rates: np.ndarray
    path: Path


def load_cookie_dataset(path: Optional[Path] = None) -> CookieDataset:
    """Load the cookie dataset from CSV and prepare tensors for Pyro models.

    If `path` is None, uses the default path from `generate_data.get_default_data_path()`.
    Returns a CookieDataset with:
      - cookies: DataFrame with columns ['chips', 'location']
      - location_stats: per-location mean/std/count
      - locations: integer-encoded per-observation location indices (torch.long)
      - chips: float tensor of chip counts (torch.float32)
      - n_locations: number of unique locations (int)
      - location_rates: per-location means as numpy array (for plotting)
      - sample_sizes: per-location counts as numpy array (for labeling)
      - path: the resolved CSV file path
    """
    csv_path = path or get_default_data_path()
    # Load from CSV if present; otherwise generate deterministically and optionally save.
    cookies = load_or_generate_cookie_data(path=csv_path, save_if_missing=True)
    cookies["location"] = pd.Categorical(cookies["location"], ordered=True)

    location_stats = (
        cookies.groupby("location", observed=False)["chips"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    le = LabelEncoder()
    locations = torch.tensor(le.fit_transform(cookies["location"]), dtype=torch.long)
    chips = torch.tensor(cookies["chips"].values, dtype=torch.float32)
    n_locations = int(cookies["location"].nunique())

    location_rates = location_stats["mean"].to_numpy()
    sample_sizes = location_stats["count"].to_numpy()
    true_rates = DEFAULT_LOCATION_RATES[:n_locations]

    return CookieDataset(
        cookies=cookies,
        location_stats=location_stats,
        locations=locations,
        chips=chips,
        n_locations=n_locations,
        location_rates=location_rates,
        sample_sizes=sample_sizes,
        true_rates=true_rates,
        path=csv_path,
    )
