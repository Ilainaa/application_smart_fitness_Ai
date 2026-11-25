"""Camera adapters and locked inference engine, shared by both live modes."""
from __future__ import annotations

from pathlib import Path
import threading
import time
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from fitness_core import (SCHEMA_VERSION, FEATURE_COLUMNS, CLASSES, LandmarkEMA,
                          ExerciseTracker, extract_features, visible)

ROOT = Path(__file__).resolve().parent


def camera_source(value):
    value = str(value).strip()
    if value.isdigit():
        return int(value)
    if value.startswith(("http://", "https://", "rtsp://")):
        return value
    raise ValueError("Camera source must be a nonnegative index or http(s)/rtsp URL")


def open_capture(source):
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("เปิดกล้องไม่ได้ ตรวจสอบ index / URL และสิทธิ์การใช้กล้อง")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def create_pose():
    return mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                 smooth_landmarks=False, min_detection_confidence=0.6,
                                 min_tracking_confidence=0.6)


def landmarks_array(landmarks):
    return np.array([[p.x, p.y, p.z, p.visibility] for p in landmarks.landmark], dtype=float)


def draw_skeleton(frame, points, valid):
    h, w = frame.shape[:2]
    color = (135, 230, 80) if valid else (85, 75, 245)
    for a, b in mp.solutions.pose.POSE_CONNECTIONS:
        if min(points[a, 3], points[b, 3]) >= 0.55:
            pa = tuple((np.clip(points[a, :2], 0, 1) * [w - 1, h - 1]).astype(int))
            pb = tuple((np.clip(points[b, :2], 0, 1) * [w - 1, h - 1]).astype(int))
            cv2.line(frame, pa, pb, color, 3, cv2.LINE_AA)
    for p in points:
        if p[3] >= 0.55:
            cv2.circle(frame, tuple((np.clip(p[:2], 0, 1) * [w - 1, h - 1]).astype(int)), 4, color, -1)


def load_model(path):
    # joblib is executable serialization: load only your locally trained trusted file.
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Model schema mismatch; retrain using 2_train_model.py")
    if artifact.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Model feature order mismatch; retrain model")
    pipeline = artifact.get("pipeline")
    if not hasattr(pipeline, "predict_proba") or not set(pipeline.classes_) <= set(CLASSES):
        raise ValueError("Invalid classifier or unknown classes")
    return artifact


class FitnessEngine:
    def __init__(self, exercise, artifact=None, confidence=0.6):
        self.lock = threading.RLock()
        self.tracker = ExerciseTracker(exercise)
        self.smoother = LandmarkEMA()
        self.model = artifact["pipeline"] if artifact else None
        self.confidence = confidence
        self.pose = None
        self.closed = False
        self.error = None
        self.last_frame_time = None

    def process(self, frame):
        with self.lock:
            if self.closed:
                return frame
            now = time.monotonic()
            if self.last_frame_time is not None and now - self.last_frame_time > 0.5:
                self.smoother.reset()
            self.last_frame_time = now
            try:
                if self.pose is None:
                    self.pose = create_pose()
                # Same maximum processing resolution in collection and live app.
                if frame.shape[1] > 960:
                    frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
                result = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.pose_landmarks is None:
                    self.smoother.reset()
                    self.tracker.invalidate(now)
                else:
                    raw = landmarks_array(result.pose_landmarks)
                    if not visible(raw, self.tracker.exercise):
                        self.smoother.reset()
                        self.tracker.invalidate(now)
                        draw_skeleton(frame, raw, False)
                    else:
                        points = self.smoother.update(raw)
                        f = extract_features(points, frame.shape[1], frame.shape[0])
                        label, probability = None, 1.0
                        if self.model is not None:
                            row = pd.DataFrame([[f[k] for k in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
                            probabilities = self.model.predict_proba(row)[0]
                            winner = int(np.argmax(probabilities))
                            label, probability = str(self.model.classes_[winner]), float(probabilities[winner])
                        self.tracker.update(f, now, label, probability, self.confidence)
                        draw_skeleton(frame, points, self.tracker.latest["valid"])
                self.error = None
            except ValueError as exc:
                self.smoother.reset()
                self.tracker.invalidate(now, "ข้อมูลท่าทางไม่สมบูรณ์ กรุณาขยับกล้อง")
                self.error = str(exc)
            except Exception as exc:
                self.tracker.invalidate(now, "ประมวลผลไม่สำเร็จ กรุณาจบเซสชันแล้วเริ่มใหม่")
                self.error = f"{type(exc).__name__}: {exc}"
            snap = {**self.tracker.latest, "exercise": self.tracker.exercise,
                    "seconds": self.tracker.seconds, "reps": self.tracker.reps}
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (22, 42, 42), -1)
            value = f"{snap['seconds']:.1f}s" if snap["exercise"] == "plank" else f"{snap['reps']} reps"
            cv2.putText(frame, f"{snap['exercise'].upper()} | {value} | Form {snap['score']:.0f}%",
                        (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 245, 220), 2)
            return frame

    def snapshot(self):
        with self.lock:
            if not self.closed and self.last_frame_time is not None and time.monotonic() - self.last_frame_time > 0.5:
                self.tracker.invalidate(time.monotonic(), "รอภาพจากกล้อง / สัญญาณภาพขาดช่วง")
            return {**self.tracker.snapshot(), "error": self.error}

    def close(self):
        with self.lock:
            self.closed = True
            if self.pose is not None:
                self.pose.close()
                self.pose = None


class CaptureWorker:
    """Capture never blocks Streamlit. Stops after browser heartbeat disappears."""
    def __init__(self, source, engine):
        self.source, self.engine = source, engine
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.error = None
        self.heartbeat = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        cap = None
        try:
            cap = open_capture(self.source)
            while not self.stop_event.is_set() and time.monotonic() - self.heartbeat < 15:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("กล้องหยุดส่งภาพ กรุณาจบเซสชันแล้วเชื่อมต่อใหม่")
                processed = self.engine.process(frame)
                with self.lock:
                    self.frame = processed
                self.stop_event.wait(0.01)
        except Exception as exc:
            self.error = str(exc)
        finally:
            if cap is not None:
                cap.release()
            self.engine.close()

    def latest(self):
        self.heartbeat = time.monotonic()
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=6)
        self.engine.close()
