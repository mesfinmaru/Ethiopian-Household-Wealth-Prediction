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


class SmokeTests(unittest.TestCase):
    def test_core_modules_import(self):
        modules = [
            "data_loader",
            "data_cleaner",
            "data_preprocesor",
            "feature_enginner",
            "modeling",
        ]
        for module_name in modules:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)

    def test_key_functions_exist(self):
        data_loader = importlib.import_module("data_loader")
        modeling = importlib.import_module("modeling")

        self.assertTrue(hasattr(data_loader, "build_all_waves"))
        self.assertTrue(hasattr(data_loader, "build_wave"))
        self.assertTrue(hasattr(modeling, "WealthPredictor"))

    def test_processed_data_exists(self):
        processed_csv = ROOT / "data" / "processed" / "all_waves_clean.csv"
        self.assertTrue(processed_csv.exists(), f"Missing processed dataset: {processed_csv}")


if __name__ == "__main__":
    unittest.main(verbosity=2)