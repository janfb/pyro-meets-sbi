"""
Cookie chip count data generation utilities.

Generates a synthetic dataset for the cookie factory example and writes it to
CSV for reuse in notebooks and scripts.

Usage:
    python -m pyro_meets_sbi.generate_data

This will create `pyro-meets-sbi/data/cookie_chips_data.csv` at the repo root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

DEFAULT_LOCATION_RATES: List[float] = [9.3, 6.0, 9.6, 8.9, 8.1]
DEFAULT_SAMPLE_SIZES: List[int] = [30, 30, 30, 30, 5]
DEFAULT_SEED: int = 42


@dataclass(frozen=True)
class CookieDataSpec:
    rates: Tuple[float, ...] = tuple(DEFAULT_LOCATION_RATES)
    sample_sizes: Tuple[int, ...] = tuple(DEFAULT_SAMPLE_SIZES)
    seed: int = DEFAULT_SEED

    def validate(self) -> None:
        if len(self.rates) != len(self.sample_sizes):
            raise ValueError("rates and sample_sizes must have same length")
        if any(s <= 0 for s in self.sample_sizes):
            raise ValueError("sample_sizes must be positive integers")
        if any(r <= 0 for r in self.rates):
            raise ValueError("rates must be positive")


def generate_cookie_data(spec: CookieDataSpec | None = None) -> pd.DataFrame:
    """Generate the cookie dataset as a pandas DataFrame.

    Columns:
      - chips (int): number of chips for each cookie
      - location (int): location id in [1..L]
    """
    spec = spec or CookieDataSpec()
    spec.validate()

    rng = np.random.default_rng(spec.seed)

    locations_list: list[int] = []
    chips_list: list[int] = []

    for loc_id, (rate, n) in enumerate(zip(spec.rates, spec.sample_sizes), start=1):
        chips_loc = rng.poisson(lam=rate, size=int(n)).astype(int).tolist()
        chips_list.extend(chips_loc)
        locations_list.extend([loc_id] * int(n))

    cookies = pd.DataFrame({"chips": chips_list, "location": locations_list})
    # Keep location as categorical with natural order 1..L
    cookies["location"] = pd.Categorical(cookies["location"], ordered=True)
    return cookies


def get_default_data_path() -> Path:
    # This file lives at repo_root/src/pyro_meets_sbi/generate_data.py
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "cookie_chips_data.csv"


def save_cookie_data(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or get_default_data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    spec = CookieDataSpec()
    df = generate_cookie_data(spec)
    out = save_cookie_data(df)
    by_loc = df.groupby("location", observed=False)["chips"].agg(
        ["count", "mean", "std"]
    )  # pandas>=2 compat
    print(f"Saved dataset to: {out}")
    print(by_loc)


if __name__ == "__main__":
    main()
