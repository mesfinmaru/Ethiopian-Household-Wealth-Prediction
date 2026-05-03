"""Functions for merging ESS sections and combining waves."""
from __future__ import annotations

from typing import Dict

import pandas as pd


def _ensure_hhid(df: pd.DataFrame) -> pd.DataFrame:
    if "hhid" in df.columns:
        return df
    candidates = [c for c in df.columns if "hhid" in c or "household" in c]
    if candidates:
        return df.rename(columns={candidates[0]: "hhid"})
    raise KeyError("No household id column found")


def merge_wave_sections(sections: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = _ensure_hhid(sections["cons_agg"]).copy()
    for key in ["sect1_hh", "sect3_hh", "sect7_hh", "sect8_hh"]:
        if key in sections:
            sdf = _ensure_hhid(sections[key]).copy()
            sdf = sdf.groupby("hhid", as_index=False).first()
            base = base.merge(sdf, on=["hhid", "year"], how="left", suffixes=("", f"_{key}"))
    return base


def combine_waves(waves: Dict[int, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    merged = [merge_wave_sections(sec) for _, sec in sorted(waves.items()) if "cons_agg" in sec]
    if not merged:
        return pd.DataFrame()
    return pd.concat(merged, ignore_index=True, sort=False)
