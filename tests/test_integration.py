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
        self.assertEqual(app.session_state["screen"], "choose")
        app.checkbox(key="rule_only").check().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.button(key="start").disabled)
        app.button(key="start").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["active"])
        self.assertEqual(app.session_state["screen"], "train")
        self.assertTrue(app.button(key="nav_review").disabled)
        self.assertTrue(app.button(key="nav_labeler").disabled)
        self.assertFalse(app.checkbox)
        webrtc_widget.assert_called()
        end_button = next(button for button in app.button if "จบเซสชัน" in button.label)
        end_button.click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.session_state["active"])
        self.assertIsNotNone(app.session_state["review"])
        self.assertEqual(app.session_state["screen"], "review")
        app.button(key="nav_train").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.checkbox(key="rule_only").value)
        app.button(key="nav_labeler").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.get("file_uploader"))
        app.button(key="nav_train").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["screen"], "choose")
        self.assertTrue(app.checkbox(key="rule_only").value)

    @patch("camera_pipeline.CaptureWorker")
    def test_capture_screen_uses_worker_and_closes_on_end(self, capture):
        from streamlit.testing.v1 import AppTest
        capture.return_value.latest.return_value = np.zeros((48, 64, 3), dtype=np.uint8)
        capture.return_value.error = None
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
        app.checkbox(key="rule_only").check().run()
        app.radio(key="camera_choice").set_value("DroidCam / IP Camera / Webcam").run()
        app.button(key="pick_plank").click().run()
        app.slider(key="ai_confidence").set_value(.75).run()
        app.button(key="start").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(capture.call_args.args[0], 0)
        capture.return_value.latest.assert_called()
        self.assertEqual(app.session_state["engine"].snapshot()["exercise"], "plank")
        app.button(key="end").click().run()
        self.assertFalse(app.exception)
        capture.return_value.stop.assert_called_once()
        self.assertTrue(app.session_state["engine"].closed)
        app.button(key="nav_train").click().run()
        self.assertEqual(app.slider(key="ai_confidence").value, .75)
        self.assertEqual(app.radio(key="camera_choice").value, "DroidCam / IP Camera / Webcam")
        self.assertEqual(app.session_state["preferences"]["exercise"], "plank")

    def test_review_grade_charts_and_export_with_recorded_data(self):
        from streamlit.testing.v1 import AppTest
        from camera_pipeline import FitnessEngine
        for exercise, score, grade in [("squat", 85, "A"), ("squat", 70, "B"), ("plank", 69, "C")]:
            with self.subTest(exercise=exercise, score=score):
                engine = FitnessEngine(exercise)
                review = engine.snapshot()
                engine.close()
                review.update(frames=10, average_score=score, reps=2, correct_reps=1, seconds=3.5,
                              history=[{"elapsed_s": i, "form_score": score, "reps": i,
                                        "correct_reps": min(i, 1), "plank_s": i * 1.75,
                                        "knee": 90, "elbow": 90, "body": 170} for i in range(3)])
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40)
                app.session_state["screen"] = "review"
                app.session_state["review"] = review
                app.session_state["session_rule_only"] = True
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(any(f'Performance Grade {grade}' in m.value for m in app.markdown))
                self.assertEqual(len(app.get("arrow_vega_lite_chart")), 2)
                self.assertEqual(len(app.get("download_button")), 1)


if __name__ == "__main__":
    unittest.main()
