"""
feature_enginner.py — Feature Engineering for ESS Wealth Prediction
Targeted domain-driven feature creation.  No blind interactions (would
create 1000+ useless features from 45 inputs).
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """
    Create domain-informed features from the cleaned ESS dataset.

    All features created are interpretable and grounded in development
    economics literature on poverty measurement in Sub-Saharan Africa.

    Usage
    -----
        fe = FeatureEngineer()
        df = fe.engineer_all(df)
        print(fe.created_features_)   # list of new column names
    """

    def __init__(self) -> None:
        self.created_features_: list[str] = []

    def engineer_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        df = df.copy()
        df = self._household_composition(df)
        df = self._housing_quality_index(df)
        df = self._asset_wealth_index(df)
        df = self._labour_intensity(df)
        df = self._vulnerability_index(df)
        df = self._geographic_features(df)
        df = self._head_human_capital(df)
        return df

    # ── 1. Household composition ──────────────────────────────────────────

    def _household_composition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Capture household structure effects on wealth.

        adults_ratio     : share of adults (proxy for productive capacity)
        is_large_hh      : 1 if hh_size >= 7 (large families often poorer)
        is_single_person : 1 if hh_size == 1 (can be urban wealthy or isolated poor)
        """
        if "hh_size" in df.columns and "adulteq" in df.columns:
            df["adults_ratio"] = (
                df["adulteq"] / df["hh_size"].replace(0, np.nan)
            ).clip(0, 1)
            df["is_large_hh"]      = (df["hh_size"] >= 7).astype(int)
            df["is_single_person"] = (df["hh_size"] == 1).astype(int)
            self.created_features_ += ["adults_ratio","is_large_hh","is_single_person"]
        return df

    # ── 2. Housing quality index ──────────────────────────────────────────

    def _housing_quality_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recode ordinal housing materials to quality scores (0=worst, 1=best).

        ESS coding for roof: 1=iron/zinc(best) → 4=grass/leaves(worst)
        ESS coding for wall: 1=concrete(best)  → varies
        ESS coding for floor: 1=earth(worst)   → higher=better (reversed)
        ESS coding for water: 1=piped(best)    → 6+=surface(worst)
        ESS coding for toilet:1=flush(best)    → 5=open field(worst)
        ESS coding for fuel:  lower=cleaner    → firewood/dung=worst

        improved_water    : 1 if piped/tube-well/protected (codes 1-4)
        improved_sanitation: 1 if flush/VIP/pit with slab (codes 1-3)
        clean_fuel        : 1 if electricity/gas/kerosene (codes 1-3 in W1)
        housing_quality   : composite 0–1 index from available components
        """
        quality_parts = []

        # Roof: 1=iron(best), higher=worse → invert
        if "roof" in df.columns:
            r = df["roof"].clip(1, 6)
            df["roof_quality"] = ((6 - r) / 5).clip(0, 1)
            quality_parts.append(df["roof_quality"])
            self.created_features_.append("roof_quality")

        # Floor: ESS W1 — 3=earth, 1=wood, 4=cement, 5=tiles → non-linear
        # Recode: 1=earth/mud=0, tiles/cement=1
        if "floor" in df.columns:
            df["floor_quality"] = df["floor"].map(
                {1: 0.5, 2: 0.5, 3: 0.0, 4: 1.0, 5: 1.0, 6: 0.25}
            ).fillna(0.25)
            quality_parts.append(df["floor_quality"])
            self.created_features_.append("floor_quality")

        # Water: 1=piped into house(best), 6+=surface(worst)
        if "water" in df.columns:
            df["improved_water"] = (df["water"] <= 4).astype(float)
            df["improved_water"][df["water"].isna()] = np.nan
            quality_parts.append(df["improved_water"])
            self.created_features_ += ["improved_water"]

        # Toilet: 1=flush(best), 5=open field(worst)
        if "toilet" in df.columns:
            df["improved_sanitation"] = (df["toilet"] <= 3).astype(float)
            df["improved_sanitation"][df["toilet"].isna()] = np.nan
            quality_parts.append(df["improved_sanitation"])
            self.created_features_.append("improved_sanitation")

        # Cooking fuel: 1-3=clean(electricity/gas/kerosene), 4+=solid fuels
        if "fuel" in df.columns:
            df["clean_fuel"] = (df["fuel"] <= 3).astype(float)
            df["clean_fuel"][df["fuel"].isna()] = np.nan
            quality_parts.append(df["clean_fuel"])
            self.created_features_.append("clean_fuel")

        # Composite housing quality index
        if quality_parts:
            df["housing_quality_idx"] = (
                pd.concat(quality_parts, axis=1).mean(axis=1)
            )
            self.created_features_.append("housing_quality_idx")

        return df

    # ── 3. Asset wealth index ─────────────────────────────────────────────

    def _asset_wealth_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Modern asset ownership index (proxy for permanent income).

        modern_asset_score : weighted sum (phone=1, TV=2, fridge=3)
          TV > phone because TVs are more expensive; fridges highest barrier
        has_any_modern_asset : 1 if owns at least one modern asset
        """
        score = pd.Series(0.0, index=df.index)
        weights = {"owns_phone": 1, "owns_tv": 2, "owns_fridge": 3}
        for col, w in weights.items():
            if col in df.columns:
                score += df[col].fillna(0) * w

        if any(c in df.columns for c in weights):
            df["modern_asset_score"]   = score
            df["has_any_modern_asset"] = (score > 0).astype(int)
            self.created_features_ += ["modern_asset_score","has_any_modern_asset"]

        return df

    # ── 4. Labour intensity ───────────────────────────────────────────────

    def _labour_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Labour market engagement features.

        labour_intensity : n_workers / hh_size (share of household employed)
        is_fully_dependent: 1 if no wage earners at all
        """
        if "hh_n_workers" in df.columns and "hh_size" in df.columns:
            df["labour_intensity"] = (
                df["hh_n_workers"] / df["hh_size"].replace(0, np.nan)
            ).fillna(0).clip(0, 1)
            self.created_features_.append("labour_intensity")

        if "hh_any_wage_earner" in df.columns:
            df["is_fully_dependent"] = (df["hh_any_wage_earner"] == 0).astype(int)
            self.created_features_.append("is_fully_dependent")

        return df

    # ── 5. Vulnerability index ────────────────────────────────────────────

    def _vulnerability_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate shock exposure into a vulnerability score.

        shock_breadth : total types of shocks experienced (0–4)
        is_multi_shock: 1 if affected by ≥2 distinct shock types
        """
        shock_cols = ["experienced_drought","experienced_illness",
                      "experienced_death","experienced_crop_loss"]
        available  = [c for c in shock_cols if c in df.columns]

        if available:
            df["shock_breadth"]  = df[available].fillna(0).sum(axis=1).astype(int)
            df["is_multi_shock"] = (df["shock_breadth"] >= 2).astype(int)
            self.created_features_ += ["shock_breadth","is_multi_shock"]

        return df

    # ── 6. Geographic features ────────────────────────────────────────────

    def _geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Geography-based wealth signals.

        is_urban          : 1 if settlement == 0 (urban)
        is_addis          : 1 if region is ADDIS ABABA (wealthiest)
        is_peripheral     : 1 if Afar/Somali/Gambela/Benishangul (remote)
        urban_conflict    : is_urban × is_tigray_conflict interaction
        """
        if "settlement" in df.columns:
            df["is_urban"] = (df["settlement"] == 0).astype(int)
            self.created_features_.append("is_urban")

        if "region" in df.columns:
            region_str = df["region"].astype(str).str.upper()
            df["is_addis"]      = (region_str == "ADDIS ABABA").astype(int)
            df["is_peripheral"] = region_str.isin(
                ["AFAR","SOMALI","GAMBELA","BENISHANGUL GUMUZ"]
            ).astype(int)
            self.created_features_ += ["is_addis","is_peripheral"]

        if "is_urban" in df.columns and "is_tigray_conflict" in df.columns:
            df["urban_conflict"] = df["is_urban"] * df["is_tigray_conflict"]
            self.created_features_.append("urban_conflict")

        return df

    # ── 7. Head human capital ─────────────────────────────────────────────

    def _head_human_capital(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Education × age interactions for household head.

        head_prime_working_age : 1 if head is 25–55 (peak productivity)
        educated_prime_head    : educated AND prime-age interaction
        head_elderly           : 1 if head ≥ 60 (retirement, lower income)
        """
        if "head_age" in df.columns:
            df["head_prime_working_age"] = (
                (df["head_age"] >= 25) & (df["head_age"] <= 55)
            ).astype(int)
            df["head_elderly"] = (df["head_age"] >= 60).astype(int)
            self.created_features_ += ["head_prime_working_age","head_elderly"]

        if "head_edu_level" in df.columns and "head_prime_working_age" in df.columns:
            df["educated_prime_head"] = (
                (df["head_edu_level"] >= 3) & (df["head_prime_working_age"] == 1)
            ).astype(int)
            self.created_features_.append("educated_prime_head")

        return df