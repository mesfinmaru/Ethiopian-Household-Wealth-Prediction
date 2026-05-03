"""Data loading utilities for Ethiopian LSMS/ESS multi-wave household welfare modeling."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import pyreadstat


STANDARD_COL_MAP = {
    "household_id": "hhid",
    "hh_id": "hhid",
    "householdid": "hhid",
    "ea_id": "ea",
    "region_code": "region",
    "region_name": "region",
    "urban_rural": "urban",
    "rural_urban": "urban",
}


class LSMSDataLoader:
    """Loader that supports both CSV and SAV sources and standardizes core columns."""

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace(r"[^a-z0-9_]", "", regex=True)
        )
        rename_map = {c: STANDARD_COL_MAP[c] for c in df.columns if c in STANDARD_COL_MAP}
        return df.rename(columns=rename_map)

    @staticmethod
    def _read_any(path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False)
        if path.suffix.lower() == ".sav":
            df, _ = pyreadstat.read_sav(str(path))
            return df
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    def load_wave_sections(
        self,
        wave_path: str | Path,
        year: int,
        sections: Iterable[str] = ("sect1_hh", "sect3_hh", "sect7_hh", "sect8_hh", "cons_agg"),
    ) -> Dict[str, pd.DataFrame]:
        wave_path = Path(wave_path)
        loaded: Dict[str, pd.DataFrame] = {}
        for section in sections:
            matches = list(wave_path.rglob(f"*{section}*.csv")) + list(wave_path.rglob(f"*{section}*.sav"))
            if not matches:
                continue
            df = self._normalize_columns(self._read_any(matches[0]))
            df["year"] = year
            loaded[section] = df
        return loaded

    def load_multi_wave(self, wave_map: Dict[int, str | Path]) -> Dict[int, Dict[str, pd.DataFrame]]:
        return {year: self.load_wave_sections(path, year) for year, path in wave_map.items()}
