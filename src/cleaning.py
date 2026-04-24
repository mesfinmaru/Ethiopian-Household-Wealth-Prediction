import pandas as pd
import pyreadstat
import os

class DataCleaner:
    def __init__(self):
        self.region_map = {
            1: 'TIGRAY', 2: 'AFAR', 3: 'AMHARA', 4: 'OROMIA', 5: 'SOMALI',
            6: 'BENISHANGUL GUMUZ', 7: 'SNNP', 12: 'GAMBELA', 13: 'HARAR',
            14: 'ADDIS ABABA', 15: 'DIRE DAWA'
        }

    def load_raw_data(self, folder_path='data/raw/'):
        """Loads all waves from the raw directory."""
        waves = {}
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                waves[file.split('.')[0]] = pd.read_csv(os.path.join(folder_path, file))
            elif file.endswith('.sav'):
                df, _ = pyreadstat.read_sav(os.path.join(folder_path, file))
                waves[file.split('.')[0]] = df
        return waves

    def standardize_regions(self, df):
        """Fixes the saq01 numeric codes vs strings."""
        if 'saq01' in df.columns:
            if df['saq01'].dtype != object:
                df['Region'] = df['saq01'].map(self.region_map)
            else:
                df['Region'] = df['saq01'].str.upper()
        return df