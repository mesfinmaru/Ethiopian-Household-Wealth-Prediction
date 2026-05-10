import importlib
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class DataPipelineTests(unittest.TestCase):
    def test_build_clean_engineer_pipeline(self):
        data_loader = importlib.import_module("data_loader")
        data_cleaner = importlib.import_module("data_cleaner")
        feature_engineer = importlib.import_module("feature_enginner")

        raw = data_loader.build_all_waves(save=False, verbose=False)
        self.assertFalse(raw.empty)

        cleaner = data_cleaner.DataCleaner()
        cleaned = cleaner.fit_transform(raw.copy())
        self.assertFalse(cleaned.empty)
        self.assertIn("has_full_housing", cleaned.columns)
        self.assertIn("has_enterprise_data", cleaned.columns)

        engineer = feature_engineer.FeatureEngineer()
        engineered = engineer.engineer_all(cleaned.copy())
        expected = ["shock_breadth", "is_multi_shock", "housing_quality_idx", "modern_asset_score"]
        for column in expected:
            with self.subTest(column=column):
                self.assertIn(column, engineered.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)