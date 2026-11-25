"""Temporary synthetic fixtures verify plumbing, NOT real-world model quality."""
from argparse import Namespace
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from fitness_core import CLASSES, FEATURE_COLUMNS, SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_model", ROOT / "2_train_model.py")
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class IntegrationTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(42)
        rows = []
        for group in range(4):
            for label_index, label in enumerate(CLASSES):
                for _ in range(10):
                    values = rng.normal(label_index * 3, .1, len(FEATURE_COLUMNS))
                    rows.append({**dict(zip(FEATURE_COLUMNS, values)), "label": label,
                                 "schema_version": SCHEMA_VERSION, "session_id": f"s{group}"})
        return pd.DataFrame(rows)

    def test_train_save_reload_and_group_isolation(self):
        from camera_pipeline import load_model
        data = self.data()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data.to_csv(root / "poses.csv", index=False)
            args = Namespace(data=root / "poses.csv", output=root / "models", group_by="session_id",
                             seed=42, test_size=.25, model="rf", max_iter=50)
            with contextlib.redirect_stdout(io.StringIO()):
                metadata = training.train(args)
            artifact = load_model(root / "models/fitness_model.pkl")
            self.assertEqual(set(artifact["pipeline"].classes_), set(CLASSES))
            self.assertTrue(set(metadata["train_groups"]).isdisjoint(metadata["test_groups"]))
            self.assertTrue((root / "models/confusion_matrix.png").is_file())
            self.assertTrue((root / "models/training_metrics.json").is_file())
            self.assertEqual(artifact["pipeline"].predict_proba(data[FEATURE_COLUMNS].iloc[:1]).shape, (1, len(CLASSES)))

    def test_insufficient_groups_fail_without_frame_split(self):
        data = self.data()
        data.session_id = "only_one"
        with self.assertRaisesRegex(ValueError, "independent groups"):
            training.split_groups(data, "session_id", .25, 42)

    def test_engine_blank_frame_and_resource_cleanup(self):
        from camera_pipeline import FitnessEngine
        engine = FitnessEngine("plank")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = engine.process(frame)
        self.assertEqual(output.shape, frame.shape)
        snapshot = engine.snapshot()
        self.assertIsNone(snapshot["error"])
        self.assertEqual(snapshot["seconds"], 0)
        self.assertFalse(snapshot["valid"])
        engine.close()
        engine.close()
        self.assertIsNone(engine.pose)

    @patch("streamlit_webrtc.webrtc_streamer")
    def test_streamlit_choose_and_review_flow(self, webrtc_widget):
        from streamlit.testing.v1 import AppTest
        # AppTest has no real websocket/session manager; mock only that boundary.
        # Browser/WebRTC transport and physical cameras need a separate live test.
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
        self.assertFalse(app.exception)
        app.sidebar.checkbox[0].check().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.button[0].disabled)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["active"])
        webrtc_widget.assert_called()
        end_button = next(button for button in app.button if "จบเซสชัน" in button.label)
        end_button.click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.session_state["active"])
        self.assertIsNotNone(app.session_state["review"])


if __name__ == "__main__":
    unittest.main()
