"""
feature_engineer.py
═══════════════════════════════════════════════════════════════════════════════
FeatureEngineer — domain-driven feature creation for Ethiopian wealth prediction.

Design principle: every feature is grounded in poverty measurement literature
for Sub-Saharan Africa. No blind polynomial/interaction expansion — each
engineering choice has an explicit economic rationale.

Feature groups created:
  1. household_composition  — adults_ratio, is_large_hh, is_single_person
  2. housing_quality_index  — improved_water/sanitation, clean_fuel, roof_quality,
                              floor_quality, housing_quality_idx
  3. asset_wealth_index     — modern_asset_score, has_any_modern_asset
  4. labour_intensity       — labour_intensity, is_fully_dependent
  5. vulnerability_index    — shock_breadth, is_multi_shock
  6. geographic_features    — is_urban, is_addis, is_peripheral, urban_conflict
  7. head_human_capital     — head_prime_working_age, head_elderly,
                              educated_prime_head
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Create interpretable domain-informed features from cleaned ESS data.

    Usage
    -----
        fe = FeatureEngineer()
        df = fe.engineer_all(df_clean)
        fe.created_features_   # list of new column names
        fe.summary()           # DataFrame: feature name + economic rationale
    """

    def __init__(self):
        self.created_features_ = []
        self._meta             = []

    def engineer_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all 7 feature engineering groups in recommended order."""
        df = self.household_composition(df)
        df = self.housing_quality_index(df)
        df = self.asset_wealth_index(df)
        df = self.labour_intensity(df)
        df = self.vulnerability_index(df)
        df = self.geographic_features(df)
        df = self.head_human_capital(df)
        return df

    def summary(self) -> pd.DataFrame:
        """Return table of engineered features with economic rationale."""
        return pd.DataFrame(self._meta)

    # ── 1. Household composition ───────────────────────────────────────────────

    def household_composition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Household structure proxies for productive capacity.

        adults_ratio     : adulteq / hh_size — share of adult-equivalent members
        is_large_hh      : 1 if hh_size ≥ 7 (larger families correlated with poverty)
        is_single_person : 1 if hh_size == 1 (elderly/isolated households)

        Economic rationale: dependency burden is a primary poverty driver in
        Sub-Saharan Africa (Foster-Greer-Thorbecke poverty decompositions).
        """
        df = df.copy()
        if "hh_size" in df.columns and "adulteq" in df.columns:
            df["adults_ratio"]     = (df["adulteq"] / df["hh_size"]
                                      .replace(0, np.nan)).clip(0, 1)
            df["is_large_hh"]      = (df["hh_size"] >= 7).astype(int)
            df["is_single_person"] = (df["hh_size"] == 1).astype(int)
            self._add("adults_ratio",     "adulteq / hh_size — productive capacity")
            self._add("is_large_hh",      "1 if hh_size ≥ 7 — high dependency")
            self._add("is_single_person", "1 if hh_size == 1 — isolated household")
        return df

    # ── 2. Housing quality index ───────────────────────────────────────────────

    def housing_quality_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recode ordinal ESS housing materials into interpretable quality scores.

        ESS coding convention: lower code = better quality (1=best, higher=worse).
        Exception: floor type — codes vary by wave, remapped explicitly.

        improved_water      : 1 if piped/tube-well/protected source (codes 1–4)
        improved_sanitation : 1 if flush/VIP/pit with slab (codes 1–3)
        clean_fuel          : 1 if electricity/gas/kerosene (codes 1–3)
        roof_quality        : 0–1 score (1=iron sheets, 0=grass/leaves)
        floor_quality       : 0–1 score (0=earth/mud, 1=tiles/cement)
        housing_quality_idx : composite mean of available indicators

        """
        df    = df.copy()
        parts = []

        if "roof" in df.columns:
            r = df["roof"].clip(1, 6)
            df["roof_quality"] = ((6 - r) / 5).clip(0, 1)
            parts.append(df["roof_quality"])
            self._add("roof_quality", "1-(code-1)/5  [1=iron sheets, 0=grass]")

        if "floor" in df.columns:
            df["floor_quality"] = df["floor"].map(
                {1:0.0, 2:0.25, 3:0.0, 4:1.0, 5:1.0, 6:0.5}
            ).fillna(0.25)
            parts.append(df["floor_quality"])
            self._add("floor_quality", "recoded: earth=0, cement/tiles=1")

        if "water" in df.columns:
            df["improved_water"] = (df["water"] <= 4).astype(float)
            df.loc[df["water"].isna(), "improved_water"] = np.nan
            parts.append(df["improved_water"])
            self._add("improved_water", "1 if piped/borehole/protected source")

        if "toilet" in df.columns:
            df["improved_sanitation"] = (df["toilet"] <= 3).astype(float)
            df.loc[df["toilet"].isna(), "improved_sanitation"] = np.nan
            parts.append(df["improved_sanitation"])
            self._add("improved_sanitation", "1 if flush/VIP/pit-with-slab")

        if "fuel" in df.columns:
            df["clean_fuel"] = (df["fuel"] <= 3).astype(float)
            df.loc[df["fuel"].isna(), "clean_fuel"] = np.nan
            parts.append(df["clean_fuel"])
            self._add("clean_fuel", "1 if electricity/gas/kerosene cooking fuel")

        if parts:
            df["housing_quality_idx"] = pd.concat(parts, axis=1).mean(axis=1)
            self._add("housing_quality_idx", "composite mean of housing quality sub-indices")

        return df

    # ── 3. Asset wealth index ──────────────────────────────────────────────────

    def asset_wealth_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Modern consumer asset ownership index (proxy for permanent income).

        Weighted sum: phone=1, TV=2, fridge=3 (reflects relative acquisition cost
        and thus wealth stratification power in Ethiopian context).

        modern_asset_score   : weighted sum 0–6
        has_any_modern_asset : 1 if owns ≥ 1 modern asset

        """
        df    = df.copy()
        score = pd.Series(0.0, index=df.index)
        for col, w in [("owns_phone", 1), ("owns_tv", 2), ("owns_fridge", 3)]:
            if col in df.columns:
                score += df[col].fillna(0) * w
        if any(c in df.columns for c in ["owns_phone","owns_tv","owns_fridge"]):
            df["modern_asset_score"]   = score
            df["has_any_modern_asset"] = (score > 0).astype(int)
            self._add("modern_asset_score",   "phone×1 + TV×2 + fridge×3")
            self._add("has_any_modern_asset", "1 if any modern consumer asset owned")
        return df

    # ── 4. Labour intensity ────────────────────────────────────────────────────

    def labour_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Labour market engagement features.

        labour_intensity   : hh_n_workers / hh_size (wage-earning share)
        is_fully_dependent : 1 if zero wage earners (no market income)

        Economic rationale: access to wage employment is the primary pathway
        out of poverty in rural Ethiopia (World Bank Ethiopia Poverty Reports).
        """
        df = df.copy()
        if "hh_n_workers" in df.columns and "hh_size" in df.columns:
            df["labour_intensity"] = (
                df["hh_n_workers"] / df["hh_size"].replace(0, np.nan)
            ).fillna(0).clip(0, 1)
            self._add("labour_intensity", "hh_n_workers / hh_size")
        if "hh_any_wage_earner" in df.columns:
            df["is_fully_dependent"] = (df["hh_any_wage_earner"] == 0).astype(int)
            self._add("is_fully_dependent", "1 if no household wage earner")
        return df

    # ── 5. Vulnerability index ─────────────────────────────────────────────────

    def vulnerability_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate shock exposure into composite vulnerability indicators.

        shock_breadth  : count of distinct shock types experienced (0–4)
        is_multi_shock : 1 if exposed to ≥ 2 distinct shock types

        Economic rationale: multiple simultaneous shocks produce non-linear
        welfare losses (covariate shocks in the Ethiopian highlands literature).
        """
        df   = df.copy()
        cols = [c for c in ["experienced_drought","experienced_illness",
                             "experienced_death","experienced_crop_loss"]
                if c in df.columns]
        if cols:
            df["shock_breadth"]  = df[cols].fillna(0).sum(axis=1).astype(int)
            df["is_multi_shock"] = (df["shock_breadth"] >= 2).astype(int)
            self._add("shock_breadth",  "count of distinct shock types (0–4)")
            self._add("is_multi_shock", "1 if ≥ 2 distinct shock types experienced")
        return df

    # ── 6. Geographic features ─────────────────────────────────────────────────

    def geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Geography-based wealth signals and regional interaction terms.

        is_urban          : 1 if settlement == 0 (urban centre)
        is_addis          : 1 if Addis Ababa (highest wealth in all waves)
        is_peripheral     : 1 if remote low-infrastructure region
                            (Afar, Somali, Gambela, Benishangul Gumuz)
        urban_conflict    : is_urban × is_tigray_conflict (W5 interaction)

        """
        df = df.copy()
        if "settlement" in df.columns:
            df["is_urban"] = (df["settlement"] == 0).astype(int)
            self._add("is_urban", "1 if urban settlement")

        if "region" in df.columns:
            rs = df["region"].astype(str).str.upper()
            df["is_addis"]      = (rs == "ADDIS ABABA").astype(int)
            df["is_peripheral"] = rs.isin(
                ["AFAR","SOMALI","GAMBELA","BENISHANGUL GUMUZ"]
            ).astype(int)
            self._add("is_addis",      "1 if Addis Ababa (richest region)")
            self._add("is_peripheral", "1 if remote/low-access region")

        if "is_urban" in df.columns and "is_tigray_conflict" in df.columns:
            df["urban_conflict"] = df["is_urban"] * df["is_tigray_conflict"]
            self._add("urban_conflict", "is_urban × is_tigray_conflict (W5 interaction)")

        return df

    # ── 7. Head human capital ──────────────────────────────────────────────────

    def head_human_capital(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Age and education interaction features for the household head.

        head_prime_working_age : 1 if head is 25–55 (peak earnings years)
        head_elderly           : 1 if head ≥ 60 (post-retirement, lower income)
        educated_prime_head    : (edu_level ≥ 3) AND prime-age interaction
                                 (education premium concentrated in prime years)

        """
        df = df.copy()
        if "head_age" in df.columns:
            df["head_prime_working_age"] = (
                (df["head_age"] >= 25) & (df["head_age"] <= 55)
            ).astype(int)
            df["head_elderly"] = (df["head_age"] >= 60).astype(int)
            self._add("head_prime_working_age", "1 if head age 25–55 (peak earnings)")
            self._add("head_elderly",           "1 if head age ≥ 60 (post-retirement)")

        if "head_edu_level" in df.columns and "head_prime_working_age" in df.columns:
            df["educated_prime_head"] = (
                (df["head_edu_level"] >= 3) & (df["head_prime_working_age"] == 1)
            ).astype(int)
            self._add("educated_prime_head",
                      "head_edu_level≥3 AND age 25–55 (education premium)")
        return df

    # ── Internal ───────────────────────────────────────────────────────────────

    def _add(self, name: str, description: str):
        """Register a new feature with its rationale."""
        if name not in self.created_features_:
            self.created_features_.append(name)
            self._meta.append({"feature": name, "description": description})
