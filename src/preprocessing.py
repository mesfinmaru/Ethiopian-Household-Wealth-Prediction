import pandas as pd

class Preprocessor:
    def rename_essential_columns(self, df):
        """Standardizes naming across all 5 waves."""
        cols = {
            'total_cons_ann': 'Annual_Spending',
            'hh_size': 'Household_Size',
            'cons_quint': 'Wealth_Rank'
        }
        return df.rename(columns=cols)

    def handle_outliers(self, df, column='Annual_Spending'):
        """Removes extreme values that skew Ethiopia's economic average."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]