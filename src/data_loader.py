"""
Ethiopian Household Survey Data Loader
=======================================
Loads and harmonizes 5 waves (2011-2022) including SPSS Wave 2 conversion.

WHY THESE 5 FILES:
- cons_agg: Total household consumption (TARGET)
- sect1_hh: Household roster → size, head age/gender (MEMBER-LEVEL → AGGREGATE)
- sect3_hh: Education → literacy, schooling (MEMBER-LEVEL → AGGREGATE)
- sect7_hh/sect8_hh: Housing quality → rooms, electricity, water (HH-LEVEL)
- sect9_hh: Asset ownership → durable goods index (HH-LEVEL)

CRITICAL: Member-level files are AGGREGATED to household level BEFORE merging
to prevent Cartesian product explosion (MemoryError).

COVID-19 & CONFLICT:
- Wave 5 (2021-22): Post-pandemic, Tigray conflict period
- 'post_covid' flag added for model to learn pandemic-specific patterns
- Regional identifiers allow conflict zone differentiation
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')


class EthiopianSurveyLoader:
    """
    Loads Ethiopian Socioeconomic Survey data across all 5 waves.
    
    Wave mapping:
    - Wave 1 (2011-12): CSV - pre-MDG baseline
    - Wave 2 (2013-14): SPSS (.sav) - growth period ~10% GDP
    - Wave 3 (2015-16): CSV - El Niño drought year
    - Wave 4 (2018-19): CSV - pre-COVID baseline
    - Wave 5 (2021-22): CSV - post-COVID + Tigray conflict
    """

    def __init__(self, base_path='../data/raw/'):
        self.base_path = base_path
        self.wave_dirs = {
            1: 'ETH_2011_ERSS_v02_M_CSV',
            2: 'ETH_2013_ESS_v03_M_SPSS',
            3: 'ETH_2015_ESS_v03_M_CSV',
            4: 'ETH_2018_ESS_v04_M_CSV',
            5: 'ETH_2021_ESPS-W5_v02_M_CSV',
        }
        self.wave_years = {1: '2011-12', 2: '2013-14', 3: '2015-16',
                           4: '2018-19', 5: '2021-22'}
        self.wave_context = {
            1: 'Pre-MDG baseline',
            2: 'High growth (~10% GDP)',
            3: 'El Niño drought',
            4: 'Pre-COVID baseline',
            5: 'Post-COVID + Tigray conflict',
        }

    # ========== FILE FINDING ==========

    def _find_file(self, wave_dir, pattern):
        """Find file matching pattern in directory (searches subdirs)"""
        matches = glob.glob(os.path.join(wave_dir, f'*{pattern}*.csv'))
        matches += glob.glob(os.path.join(wave_dir, f'*{pattern}*.CSV'))
        matches += glob.glob(os.path.join(wave_dir, '**', f'*{pattern}*.csv'), recursive=True)
        return matches[0] if matches else None

    def _find_sav(self, wave_dir, pattern):
        """Find SPSS file"""
        matches = glob.glob(os.path.join(wave_dir, f'*{pattern}*.sav'))
        matches += glob.glob(os.path.join(wave_dir, f'*{pattern}*.SAV'))
        return matches[0] if matches else None

    # ========== COLUMN HELPERS ==========

    def _get_hhid(self, df):
        """Find household ID column"""
        for c in df.columns:
            if any(kw in c.lower() for kw in ['hhid', 'household_id', 'hh_id']):
                return c
        return None

    def _clean_cols(self, df, wave):
        """Standardize column names"""
        df = df.copy()
        df.columns = (df.columns.astype(str).str.strip().str.lower()
                       .str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True))
        df['wave'] = wave
        return df

    # ========== LOAD SECTION FILE ==========

    def _load_section(self, wave, section):
        """Load one section for a wave (handles CSV and SPSS)"""
        wave_dir = os.path.join(self.base_path, self.wave_dirs[wave])
        if not os.path.exists(wave_dir):
            return None

        if wave == 2:
            # SPSS format
            fp = self._find_sav(wave_dir, section)
            if fp is None:
                return None
            import pyreadstat
            df, _ = pyreadstat.read_sav(fp)
        else:
            # CSV format
            fp = self._find_file(wave_dir, section)
            if fp is None:
                return None
            df = pd.read_csv(fp, low_memory=False)

        return self._clean_cols(df, wave)

    # ========== AGGREGATE MEMBER → HOUSEHOLD ==========

    def _agg_roster(self, df):
        """
        Aggregate sect1_hh (member-level roster) to household level.
        Returns ONE row per household with:
        - hh_size: Number of members
        - head_age: Age of household head
        - head_gender: Gender of head (0=female, 1=male)
        """
        if df is None:
            return None
        hhid = self._get_hhid(df)
        if hhid is None:
            return None

        # Count household size
        hh_size = df.groupby(hhid).size().reset_index()
        hh_size.columns = ['hhid', 'hh_size']

        # Find household head (relationship code 0 or 1)
        rel_cols = [c for c in df.columns if 'relation' in c.lower()]
        head_mask = pd.Series(True, index=df.index)
        if rel_cols:
            try:
                vals = pd.to_numeric(df[rel_cols[0]], errors='coerce')
                head_mask = vals.isin([0, 1])
            except:
                pass

        head = df[head_mask].groupby(hhid).first().reset_index()

        # Extract head age & gender
        age_cols = [c for c in head.columns if 'age' in c.lower()]
        sex_cols = [c for c in head.columns if 'sex' in c.lower() or 'gender' in c.lower()]

        keep = [hhid] + age_cols[:1] + sex_cols[:1]
        head = head[[c for c in keep if c in head.columns]]
        head = head.rename(columns={hhid: 'hhid'})
        for c in head.columns:
            if 'age' in c.lower():
                head = head.rename(columns={c: 'head_age'})
            if 'sex' in c.lower() or 'gender' in c.lower():
                head = head.rename(columns={c: 'head_gender'})

        result = hh_size.merge(head, on='hhid', how='left')

        # Encode gender if string
        if 'head_gender' in result.columns and result['head_gender'].dtype == 'object':
            result['head_gender'] = (result['head_gender'].str.lower()
                                     .map({'male': 1, 'm': 1, 'female': 0, 'f': 0}))
            result['head_gender'] = result['head_gender'].fillna(0).astype(int)

        return result

    def _agg_education(self, df):
        """
        Aggregate sect3_hh (member-level education) to household level.
        
        Returns:
        - edu_max_*: Maximum education in household (human capital ceiling)
        - edu_ratio: Proportion of members with any education (human capital breadth)
        """
        if df is None:
            return None
        hhid = self._get_hhid(df)
        if hhid is None:
            return None

        edu_kw = ['educ', 'school', 'grade', 'literacy', 'read', 'write', 'train', 'level']
        edu_cols = [c for c in df.columns if any(kw in c.lower() for kw in edu_kw)]
        num_cols = df[edu_cols].select_dtypes(include=[np.number]).columns.tolist()

        if not num_cols:
            return None

        # Max education per household
        edu_max = df.groupby(hhid)[num_cols].max().reset_index()
        edu_max = edu_max.rename(columns={hhid: 'hhid'})
        edu_max.columns = ['hhid'] + [f'edu_max_{c}' for c in edu_max.columns if c != 'hhid']

        # Proportion with any education
        total = df.groupby(hhid).size()
        has_edu = (df[num_cols].fillna(0) > 0).any(axis=1)
        edu_count = df[has_edu].groupby(hhid).size()
        edu_ratio = (edu_count / total).reset_index()
        edu_ratio.columns = ['hhid', 'edu_ratio']
        edu_ratio['edu_ratio'] = edu_ratio['edu_ratio'].fillna(0)

        return edu_max.merge(edu_ratio, on='hhid', how='left')

    # ========== MAIN BUILD METHOD ==========

    def build_dataset(self, wave):
        """
        Build complete household-level dataset for ONE wave.
        Merges 5 essential files into ~25-35 features.
        """
        print(f"\n{'='*60}")
        print(f"Wave {wave} ({self.wave_years[wave]}): {self.wave_context[wave]}")
        print(f"{'='*60}")

        # [1/5] Consumption (TARGET)
        cons = self._load_section(wave, 'cons_agg')
        if cons is None:
            print("  ✗ No consumption data - skipping wave")
            return None

        hhid = self._get_hhid(cons)
        if hhid and hhid != 'hhid':
            cons = cons.rename(columns={hhid: 'hhid'})

        # Find total consumption column
        target = None
        for c in cons.columns:
            if ('total' in c.lower() or 'aggregate' in c.lower()) and 'cons' in c.lower():
                target = c
                break
        if target is None:
            cons_cols = [c for c in cons.columns if 'cons' in c.lower()]
            target = max(cons_cols, key=lambda c: cons[c].notna().sum()) if cons_cols else None

        keep = ['hhid', 'wave', target] if target else ['hhid', 'wave']
        df = cons[[c for c in keep if c in cons.columns]].copy()
        if target:
            df = df.rename(columns={target: 'total_consumption'})
        print(f"  [1/5] Consumption: {len(df)} households")

        # [2/5] Household Roster (member → household)
        roster = self._load_section(wave, 'sect1_hh')
        roster_agg = self._agg_roster(roster)
        if roster_agg is not None:
            # Avoid duplicate columns
            dup = [c for c in roster_agg.columns if c in df.columns and c != 'hhid']
            if dup:
                roster_agg = roster_agg.drop(columns=dup)
            df = df.merge(roster_agg, on='hhid', how='left')
            print(f"  [2/5] Roster → {df.shape}")

        # [3/5] Education (member → household)
        edu = self._load_section(wave, 'sect3_hh')
        edu_agg = self._agg_education(edu)
        if edu_agg is not None:
            dup = [c for c in edu_agg.columns if c in df.columns and c != 'hhid']
            if dup:
                edu_agg = edu_agg.drop(columns=dup)
            df = df.merge(edu_agg, on='hhid', how='left')
            print(f"  [3/5] Education → {df.shape}")

        # [4/5] Housing (household-level, try sect7 then sect8)
        housing = self._load_section(wave, 'sect7_hh')
        if housing is None:
            housing = self._load_section(wave, 'sect8_hh')
        if housing is not None:
            hid = self._get_hhid(housing)
            if hid and hid != 'hhid':
                housing = housing.rename(columns={hid: 'hhid'})
            house_kw = ['room', 'electric', 'water', 'toilet', 'floor', 'wall', 'roof',
                        'dwelling', 'tenure', 'kitchen']
            keep_h = ['hhid'] + [c for c in housing.columns 
                                 if any(kw in c.lower() for kw in house_kw)]
            housing = housing[[c for c in keep_h if c in housing.columns]]
            dup = [c for c in housing.columns if c in df.columns and c != 'hhid']
            if dup:
                housing = housing.drop(columns=dup)
            df = df.merge(housing, on='hhid', how='left')
            print(f"  [4/5] Housing → {df.shape}")

        # [5/5] Assets (household-level)
        assets = self._load_section(wave, 'sect9_hh')
        if assets is not None:
            aid = self._get_hhid(assets)
            if aid and aid != 'hhid':
                assets = assets.rename(columns={aid: 'hhid'})
            asset_kw = ['own', 'radio', 'tv', 'mobile', 'phone', 'refrigerator',
                        'bicycle', 'motorcycle', 'car', 'computer', 'satellite',
                        'solar', 'stove', 'bed', 'table', 'chair']
            keep_a = ['hhid'] + [c for c in assets.columns 
                                 if any(kw in c.lower() for kw in asset_kw)
                                 and assets[c].nunique() <= 5]
            assets = assets[[c for c in keep_a if c in assets.columns]]
            dup = [c for c in assets.columns if c in df.columns and c != 'hhid']
            if dup:
                assets = assets.drop(columns=dup)
            df = df.merge(assets, on='hhid', how='left')
            # Create asset count
            asset_bin = [c for c in assets.columns if c != 'hhid']
            if asset_bin:
                df['asset_count'] = df[asset_bin].fillna(0).gt(0).sum(axis=1)
            print(f"  [5/5] Assets → {df.shape}")

        # ========== ENGINEERED FEATURES ==========

        # COVID-19 flag
        df['post_covid'] = (df['wave'] == 5).astype(int)

        # Head age categories
        if 'head_age' in df.columns:
            df['head_age'] = pd.to_numeric(df['head_age'], errors='coerce')
            df['head_elderly'] = (df['head_age'] >= 60).astype(int)
            df['head_young'] = (df['head_age'] <= 25).astype(int)

        # Log consumption (target for regression)
        if 'total_consumption' in df.columns:
            df['log_total_consumption'] = np.log1p(df['total_consumption'])
            # Per capita
            if 'hh_size' in df.columns:
                df['cons_per_capita'] = df['total_consumption'] / df['hh_size'].clip(lower=1)
                df['log_cons_per_capita'] = np.log1p(df['cons_per_capita'])

        # Gender encode if string
        if 'head_gender' in df.columns and df['head_gender'].dtype == 'object':
            df['head_gender'] = (df['head_gender'].str.lower()
                                 .map({'male': 1, 'm': 1, 'female': 0, 'f': 0}))
            df['head_gender'] = df['head_gender'].fillna(0).astype(int)

        # Cleanup
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, ~df.columns.duplicated()]
        for c in list(df.columns):
            if df[c].nunique() <= 1 and c not in ['wave', 'post_covid']:
                df = df.drop(columns=[c])

        print(f"  ✓ Complete: {df.shape[0]} HH × {df.shape[1]} features")
        return df

    def build_all_waves(self, output_dir='../data/processed/'):
        """Build and save all 5 waves, return combined DataFrame"""
        os.makedirs(output_dir, exist_ok=True)
        all_dfs = []

        for wave in range(1, 6):
            try:
                df = self.build_dataset(wave)
                if df is not None:
                    all_dfs.append(df)
                    df.to_csv(f'{output_dir}wave{wave}_clean.csv', index=False)
                    print(f"  ✓ Saved: wave{wave}_clean.csv ({len(df)} HH)")
            except Exception as e:
                print(f"  ✗ Wave {wave} ERROR: {e}")
                import traceback
                traceback.print_exc()

        if not all_dfs:
            print("\n✗ No waves loaded!")
            return None

        combined = pd.concat(all_dfs, ignore_index=True, sort=False)
        combined.to_csv(f'{output_dir}all_waves_clean.csv', index=False)

        print(f"\n{'='*60}")
        print(f"✓ FINAL DATASET: {len(combined):,} households × {combined.shape[1]} features")
        print(f"  Waves: {sorted(combined['wave'].unique())}")
        print(f"  Saved: {output_dir}all_waves_clean.csv")
        print(f"{'='*60}")

        return combined

    def get_summary(self):
        """Check data availability"""
        rows = []
        for w in range(1, 6):
            d = os.path.join(self.base_path, self.wave_dirs[w])
            rows.append({
                'Wave': w, 'Year': self.wave_years[w],
                'Exists': os.path.exists(d),
                'Context': self.wave_context[w]
            })
        return pd.DataFrame(rows)