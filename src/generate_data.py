"""
Cookie chip count data generation utilities.

Generates a synthetic dataset for the cookie factory example and writes it to
CSV for reuse in notebooks and scripts.

Usage:
    python src/generate_data.py

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
    """Return the default CSV path relative to this file.

    Works as long as this module is imported from the cloned repo, e.g., in Colab
    after `%cd /content/pyro-meets-sbi`. It simply resolves
    `../data/cookie_chips_data.csv` from the directory containing this file.
    """
    file_dir = Path(__file__).resolve().parent
    return file_dir.parent / "data" / "cookie_chips_data.csv"


def save_cookie_data(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or get_default_data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_or_generate_cookie_data(
    path: Path | None = None,
    spec: CookieDataSpec | None = None,
    *,
    save_if_missing: bool = False,
) -> pd.DataFrame:
    """Load the cookie dataset if present, otherwise generate it.

    Args:
        path: Optional path to CSV. Defaults to ``get_default_data_path()``.
        spec: Optional data specification (rates, sample sizes, seed).
        save_if_missing: If True, persist generated data to ``path``.

    Returns:
        The cookie dataset as a pandas DataFrame with columns ``chips`` (int) and
        ``location`` (categorical ordered 1..L).
    """
    path = path or get_default_data_path()

    try:
        df = pd.read_csv(path)
        # Coerce dtypes and categories for robustness
        if "chips" in df.columns:
            df["chips"] = df["chips"].astype(int)
        if "location" in df.columns:
            df["location"] = pd.Categorical(df["location"].astype(int), ordered=True)
        return df
    except FileNotFoundError:
        spec = spec or CookieDataSpec()
        df = generate_cookie_data(spec)
        if save_if_missing:
            try:
                save_cookie_data(df, path)
            except Exception:
                # Non-fatal: return in-memory data even if saving fails
                pass
        return df


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
