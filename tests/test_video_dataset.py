"""Actual encoded video fixtures + deterministic fake poses; no real subject data."""
from fractions import Fraction
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import numpy as np
import pandas as pd

from fitness_core import CSV_COLUMNS
from video_dataset import (import_video, load_project, update_clip, validate_segments, recording_id,
                           video_frames, preview_frame, export_dataset, clip_path)


class FakePose:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def process(self, frame):
        rng = np.random.default_rng(5)
        p = rng.uniform(.25, .75, (33, 4))
        p[:, 3] = .99
        p[11, :2], p[12, :2] = [.35, .25], [.65, .25]
        p[23, :2], p[24, :2] = [.4, .55], [.6, .55]
        points = [SimpleNamespace(x=x, y=y, z=z, visibility=v) for x, y, z, v in p]
        return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=points))


class EmptyPose(FakePose):
    def process(self, frame):
        return SimpleNamespace(pose_landmarks=None)


class VideoTests(unittest.TestCase):
    def setUp(self):
        # Windows sandbox TEMP can exceed MAX_PATH once a content hash is added.
        self.workspace = Path(__file__).resolve().parents[1]
        self.temp = tempfile.TemporaryDirectory(dir=self.workspace, prefix=".video-test-")
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.output = self.root / "export.csv"
        self.video = self.root / "input.mp4"
        self.make_video(self.video)

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.workspace))
        self.temp.cleanup()

    def make_video(self, path, vfr=False):
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width, stream.height = 64, 48
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, 1000)
            for index in range(20):
                frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 10, dtype=np.uint8), format="bgr24")
                frame.pts = index * 100 + (300 if vfr and index >= 10 else 0)
                frame.time_base = Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def imported(self):
        with self.video.open("rb") as source:
            clip, added = import_video(source, "phone.mp4", "p01", "take01", self.project)
        self.assertTrue(added)
        return clip

    def annotated(self):
        clip = self.imported()
        project = load_project(self.project)
        update_clip(self.project, clip["id"], project["revision"], segments=[
            {"start": 0, "end": .5, "label": "squat_up"},
            {"start": 1, "end": 1.5, "label": "squat_down_correct"}])
        return clip

    def test_import_resume_and_duplicate_identity(self):
        clip = self.imported()
        with self.video.open("rb") as source:
            duplicate, added = import_video(source, "renamed.mov", "different", "take99", self.project)
        self.assertFalse(added)
        self.assertEqual(duplicate["id"], clip["id"])
        restored = load_project(self.project)
        self.assertEqual(len(restored["clips"]), 1)
        self.assertEqual(restored["clips"][0]["subject_id"], "p01")
        self.assertTrue(clip_path(self.project, clip).exists())

    def test_segments_validation_and_adjacent_boundary(self):
        with self.assertRaisesRegex(ValueError, "ทับกัน"):
            validate_segments([{"start": 0, "end": 1, "label": "squat_up"},
                               {"start": .5, "end": 2, "label": "squat_down_wrong"}], 2)
        for start, end in [(1, 1), (-1, 1), (0, 3), (float("nan"), 1)]:
            with self.assertRaises(ValueError):
                validate_segments([{"start": start, "end": end, "label": "squat_up"}], 2)
        self.assertEqual(len(validate_segments([{"start": 0, "end": 1, "label": "squat_up"},
                                               {"start": 1, "end": 2, "label": "squat_down_wrong"}], 2)), 2)

    def test_stale_editor_rejected_and_labels_persist(self):
        clip = self.annotated()
        with self.assertRaisesRegex(ValueError, "อีกหน้าต่าง"):
            update_clip(self.project, clip["id"], 1, segments=[])
        self.assertEqual(len(load_project(self.project)["clips"][0]["segments"]), 2)

    def test_recording_groups_shared_across_clips_but_not_people(self):
        first = {"subject_id": "p1", "recording": "take01"}
        self.assertEqual(recording_id(first), recording_id(dict(first, id="different_file")))
        self.assertNotEqual(recording_id(first), recording_id(dict(first, subject_id="p2")))

    def test_video_timestamps_rotation_and_preview(self):
        frames = list(video_frames(self.video))
        self.assertEqual(len(frames), 20)
        self.assertAlmostEqual(frames[7][0], .7, places=3)
        timestamp, frame = preview_frame(self.video, .7, 90)
        self.assertAlmostEqual(timestamp, .7, places=3)
        self.assertEqual(frame.shape[:2], (64, 48))
        vfr = self.root / "variable.mp4"
        self.make_video(vfr, vfr=True)
        timestamps = [t for t, _ in video_frames(vfr)]
        self.assertAlmostEqual(timestamps[10] - timestamps[9], .4, places=3)

    def test_export_schema_gaps_labels_no_duplicates_and_group_split(self):
        self.annotated()
        report = export_dataset(self.project, self.output, 5, pose_factory=FakePose)
        frame = pd.read_csv(self.output)
        self.assertEqual(list(frame.columns), CSV_COLUMNS)
        self.assertEqual(report["rows"], 6)
        self.assertEqual(frame.label.tolist(), ["squat_up"] * 3 + ["squat_down_correct"] * 3)
        self.assertEqual(frame.session_id.nunique(), 1)
        self.assertTrue(all("video:" in t for t in frame.timestamp))
        before = self.output.read_bytes()
        export_dataset(self.project, self.output, 5, pose_factory=FakePose)
        self.assertEqual(self.output.read_bytes(), before)
        # Add an independent take with both classes: real trainer accepts the CSV.
        other = self.root / "other.mp4"
        self.make_video(other, vfr=True)
        with other.open("rb") as source:
            second, _ = import_video(source, other.name, "p01", "take02", self.project)
        data = load_project(self.project)
        training_segments = [{"start": 0, "end": .8, "label": "squat_up"},
                             {"start": 1, "end": 2, "label": "squat_down_correct"}]
        for item in data["clips"]:
            current = load_project(self.project)
            update_clip(self.project, item["id"], current["revision"], segments=training_segments)
        export_dataset(self.project, self.output, 30, pose_factory=FakePose)
        from test_integration import training
        from argparse import Namespace
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            metadata = training.train(Namespace(data=self.output, output=self.root / "models", group_by="session_id",
                                               seed=42, test_size=.5, model="rf", max_iter=50))
        self.assertTrue(set(metadata["train_groups"]).isdisjoint(metadata["test_groups"]))

    def test_empty_or_corrupt_export_preserves_previous_csv(self):
        clip = self.annotated()
        self.output.write_bytes(b"previous dataset")
        with self.assertRaisesRegex(ValueError, "ไม่พบเฟรม"):
            export_dataset(self.project, self.output, pose_factory=EmptyPose)
        self.assertEqual(self.output.read_bytes(), b"previous dataset")
        clip_path(self.project, clip).write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "ต้นฉบับเปลี่ยน"):
            export_dataset(self.project, self.output, pose_factory=FakePose)
        self.assertEqual(self.output.read_bytes(), b"previous dataset")
        self.assertFalse((self.project / ".write.lock").exists())

    def test_labeler_page_edits_and_restores_annotations(self):
        from streamlit.testing.v1 import AppTest
        self.imported()
        main = Path(__file__).resolve().parents[1] / "app.py"
        with patch("video_dataset.DEFAULT_PROJECT", self.project), patch("video_dataset.DEFAULT_OUTPUT", self.output):
            app = AppTest.from_file(str(main), default_timeout=30).run().switch_page("pages/1_Video_Labeler.py").run()
            self.assertFalse(app.exception)
            button = next(b for b in app.button if b.label == "เพิ่มช่วงและบันทึกป้ายกำกับ")
            button.click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(load_project(self.project)["clips"][0]["segments"]), 1)
            # Restarting the page must restore persisted labels without session state.
            restored = AppTest.from_file(str(main), default_timeout=30).run().switch_page("pages/1_Video_Labeler.py").run()
            self.assertFalse(restored.exception)
            self.assertTrue(any("แก้ไขหรือลบ" in e.label for e in restored.expander))

    def test_frame_generation_batch_labels_and_exact_export(self):
        from frame_labeling import generate_frames, label_frames, review_image
        clip = self.annotated()
        count = generate_frames(self.project, clip["id"], load_project(self.project)["revision"],
                                "squat", .5, pose_factory=FakePose)
        self.assertEqual(count, 4)
        data = load_project(self.project)
        item = data["clips"][0]
        frames = item["frame_set"]["frames"]
        self.assertEqual([f["time"] for f in frames], [0., .5, 1., 1.5])
        self.assertEqual(len(item["segments"]), 2)
        self.assertEqual(review_image(self.project, frames[0]).shape, (48, 64, 3))
        label_frames(self.project, clip["id"], data["revision"], [frames[0]["index"], frames[1]["index"]], "squat_up")
        label_frames(self.project, clip["id"], load_project(self.project)["revision"], [frames[2]["index"]], "skip")
        report = export_dataset(self.project, self.output)
        self.assertEqual(report["rows"], 2)
        rows = pd.read_csv(self.output)
        self.assertEqual(list(rows.columns), CSV_COLUMNS)
        self.assertEqual(rows.session_id.nunique(), 1)
        self.assertEqual(rows.label.tolist(), ["squat_up", "squat_up"])
        before = self.output.read_bytes()
        export_dataset(self.project, self.output)
        self.assertEqual(self.output.read_bytes(), before)
        update_clip(self.project, clip["id"], load_project(self.project)["revision"], annotation_mode="segments")
        self.assertEqual(export_dataset(self.project, self.output, pose_factory=FakePose)["rows"], 6)

    def test_frame_labels_reject_invalid_pose_wrong_class_stale_rotation(self):
        from frame_labeling import generate_frames, label_frames
        clip = self.imported()
        generate_frames(self.project, clip["id"], 1, "squat", pose_factory=EmptyPose)
        with self.assertRaises(ValueError):
            label_frames(self.project, clip["id"], 2, [0], "squat_up")
        with self.assertRaises(ValueError):
            label_frames(self.project, clip["id"], 2, [0], "plank_correct")
        label_frames(self.project, clip["id"], 2, [0], "skip")
        with self.assertRaises(ValueError):
            label_frames(self.project, clip["id"], 2, [0], None)
        generate_frames(self.project, clip["id"], 3, "squat", pose_factory=FakePose)
        label_frames(self.project, clip["id"], 4, [0], "squat_up")
        update_clip(self.project, clip["id"], 5, rotation=90)
        self.output.write_bytes(b"keep previous")
        with self.assertRaises(ValueError):
            export_dataset(self.project, self.output)
        self.assertEqual(self.output.read_bytes(), b"keep previous")

    def test_frame_ui_batch_save_and_reload(self):
        from frame_labeling import generate_frames
        from streamlit.testing.v1 import AppTest
        clip = self.imported()
        generate_frames(self.project, clip["id"], 1, "squat", pose_factory=FakePose)
        main = self.workspace / "app.py"
        with patch("video_dataset.DEFAULT_PROJECT", self.project), patch("video_dataset.DEFAULT_OUTPUT", self.output):
            app = AppTest.from_file(str(main), default_timeout=30).run().switch_page("pages/1_Video_Labeler.py").run()
            self.assertFalse(app.exception)
            for checkbox in app.checkbox:
                if checkbox.label.startswith("เลือกภาพ "):
                    checkbox.check()
            next(b for b in app.button if b.label == "บันทึกป้ายภาพที่เลือก").click().run()
            self.assertFalse(app.exception)
            frames = load_project(self.project)["clips"][0]["frame_set"]["frames"]
            self.assertEqual([f["label"] for f in frames], ["squat_up", "squat_up"])
            app.radio[0].set_value("segments").run()
            self.assertFalse(app.exception)
            self.assertEqual(load_project(self.project)["clips"][0]["annotation_mode"], "segments")
            self.assertEqual(len(load_project(self.project)["clips"][0]["frame_set"]["frames"]), 2)


    def test_clean_layout_pagination_and_saved_labels(self):
        from frame_labeling import generate_frames
        from streamlit.testing.v1 import AppTest
        clip = self.imported()
        generate_frames(self.project, clip["id"], 1, "squat", .1, pose_factory=FakePose)
        with patch("video_dataset.DEFAULT_PROJECT", self.project), patch("video_dataset.DEFAULT_OUTPUT", self.output):
            app = AppTest.from_file(str(self.workspace / "app.py"), default_timeout=30).run().switch_page("pages/1_Video_Labeler.py").run()
            self.assertFalse(app.exception)
            self.assertEqual([tab.label for tab in app.tabs], ["ตรวจและติดป้าย", "เพิ่มคลิป", "ส่งออกและเทรน"])
            self.assertEqual(sum(c.label.startswith("เลือกภาพ ") for c in app.checkbox), 6)
            self.assertTrue(next(b for b in app.button if b.label == "← หน้าก่อน").disabled)
            next(b for b in app.button if b.label == "หน้าถัดไป →").click().run()
            self.assertFalse(app.exception)
            next(c for c in app.checkbox if c.label == "เลือกภาพ 6").check()
            next(b for b in app.button if b.label == "บันทึกป้ายภาพที่เลือก").click().run()
            self.assertFalse(app.exception)
            frames = load_project(self.project)["clips"][0]["frame_set"]["frames"]
            self.assertEqual(frames[6]["label"], "squat_up")
            self.assertIsNone(frames[0]["label"])
            page_input = next(n for n in app.number_input if n.label.startswith("หน้าภาพ"))
            self.assertEqual(page_input.value, 2)
            page_input.set_value(4).run()
            self.assertEqual(sum(c.label.startswith("เลือกภาพ ") for c in app.checkbox), 2)
            self.assertTrue(next(b for b in app.button if b.label == "หน้าถัดไป →").disabled)


if __name__ == "__main__":
    unittest.main()
