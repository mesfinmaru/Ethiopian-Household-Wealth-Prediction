import importlib
import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class ModelPipelineTests(unittest.TestCase):
    def test_saved_model_can_predict(self):
        preprocessor_path = ROOT / "models" / "preprocessor.pkl"
        model_path = ROOT / "models" / "best_model.pkl"

        self.assertTrue(preprocessor_path.exists(), f"Missing preprocessor: {preprocessor_path}")
        self.assertTrue(model_path.exists(), f"Missing model: {model_path}")

        pipeline_, _, feature_names = joblib.load(preprocessor_path)
        model = joblib.load(model_path)

        sample = pd.DataFrame([{name: 0.0 for name in feature_names}], dtype=object)
        if "region" in sample.columns:
            sample.loc[0, "region"] = "OROMIA"
        if "settlement" in sample.columns:
            sample.loc[0, "settlement"] = 1
        if "head_edu_level" in sample.columns:
            sample.loc[0, "head_edu_level"] = 3
        if "wave" in sample.columns:
            sample.loc[0, "wave"] = 5
        for column in ["roof", "wall", "floor", "water", "toilet", "fuel"]:
            if column in sample.columns:
                sample.loc[0, column] = 1

        transformed = pipeline_.transform(sample)
        prediction = int(model.predict(transformed)[0])

        self.assertIn(prediction, [1, 2, 3, 4, 5])
        self.assertFalse(np.isnan(prediction))


if __name__ == "__main__":
    unittest.main(verbosity=2)