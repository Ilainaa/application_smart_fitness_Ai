"""Shared feature contract, geometry, smoothing and exercise state machines.

No camera/Streamlit dependency: collection, training and inference use this schema.
Angles use aspect-corrected image coordinates, NOT distorted normalized x/y.
"""
from __future__ import annotations

from collections import deque
import math
import numpy as np

SCHEMA_VERSION = 1
EXERCISES = {"squat": "Squat", "pushup": "Push-up", "plank": "Plank", "jj": "Jumping Jacks"}
CLASS_KEYS = {
    "1": "squat_up", "2": "squat_down_correct", "3": "squat_down_wrong",
    "4": "pushup_up", "5": "pushup_down_correct", "6": "pushup_down_wrong",
    "7": "plank_correct", "8": "plank_wrong", "9": "jj_closed",
    "0": "jj_open_correct", "j": "jj_open_wrong",
}
CLASSES = list(CLASS_KEYS.values())
RAW_COLUMNS = [f"raw_{i}_{axis}" for i in range(33) for axis in ("x", "y", "z", "visibility")]
LANDMARK_COLUMNS = [f"lm_{i}_{axis}" for i in range(33) for axis in ("x", "y", "z", "visibility")]
METRIC_COLUMNS = ["knee_left", "knee_right", "elbow_left", "elbow_right", "back_left", "back_right",
                  "body_left", "body_right", "shoulder_left", "shoulder_right", "ankle_ratio",
                  "wrists_above_head", "torso_horizontal", "knee", "elbow", "back", "body"]
FEATURE_COLUMNS = LANDMARK_COLUMNS + METRIC_COLUMNS
CSV_COLUMNS = ["schema_version", "session_id", "subject_id", "timestamp", "label"] + RAW_COLUMNS + FEATURE_COLUMNS


def angle(a, b, c):
    u, v = np.asarray(a) - b, np.asarray(c) - b
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    if norm < 1e-8:
        raise ValueError("Degenerate joint geometry")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / norm, -1, 1))))


class LandmarkEMA:
    def __init__(self, alpha=0.45):
        self.alpha = alpha
        self.value = None

    def reset(self):
        self.value = None

    def update(self, points):
        points = np.asarray(points, dtype=float)
        if self.value is None:
            self.value = points.copy()
        else:
            self.value[:, :3] = self.alpha * points[:, :3] + (1 - self.alpha) * self.value[:, :3]
            # Visibility must reflect the CURRENT frame, never linger after occlusion.
            self.value[:, 3] = points[:, 3]
        return self.value.copy()


def visible(points, exercise, threshold=0.55):
    if points.shape != (33, 4) or not np.isfinite(points).all():
        return False
    def good(ids):
        p = points[ids]
        return bool(np.all(p[:, 3] >= threshold) and np.all((p[:, :2] >= 0) & (p[:, :2] <= 1)))
    if exercise == "jj":
        return good([0, 11, 12, 15, 16, 23, 24, 27, 28])
    return good([7, 11, 13, 15, 23, 25, 27]) or good([8, 12, 14, 16, 24, 26, 28])


def extract_features(points, width, height):
    p = np.asarray(points, dtype=float)
    if p.shape != (33, 4) or not np.isfinite(p).all() or min(width, height) <= 0:
        raise ValueError("Invalid landmarks/frame dimensions")
    xyz = p[:, :3] * np.array([width, height, width])
    xy = xyz[:, :2]
    hips = (xyz[23] + xyz[24]) / 2
    shoulders = (xyz[11] + xyz[12]) / 2
    scale = np.linalg.norm(shoulders[:2] - hips[:2])
    shoulder_width = np.linalg.norm(xy[11] - xy[12])
    if scale < 2 or shoulder_width < 1:
        raise ValueError("Person too small or shoulder geometry unavailable")
    normalized = np.column_stack(((xyz - hips) / scale, p[:, 3]))
    result = dict(zip(LANDMARK_COLUMNS, normalized.flatten().tolist()))
    for side, ear, sh, el, wr, hip, knee, ankle in [
        ("left", 7, 11, 13, 15, 23, 25, 27), ("right", 8, 12, 14, 16, 24, 26, 28)
    ]:
        for name, ids in {"knee": (hip, knee, ankle), "elbow": (sh, el, wr),
                          "back": (ear, sh, hip), "body": (sh, hip, ankle),
                          "shoulder": (hip, sh, wr)}.items():
            result[f"{name}_{side}"] = angle(*(xy[i] for i in ids))
    # Side view: use the fully visible side with the strongest weakest landmark.
    def side_quality(ids):
        side_points = p[ids]
        if not np.all((side_points[:, :2] >= 0) & (side_points[:, :2] <= 1)):
            return -1.0
        return float(min(side_points[:, 3]))
    left = side_quality([7, 11, 13, 15, 23, 25, 27])
    right = side_quality([8, 12, 14, 16, 24, 26, 28])
    side = "left" if left >= right else "right"
    for name in ("knee", "elbow", "back", "body"):
        result[name] = result[f"{name}_{side}"]
    result["ankle_ratio"] = float(np.linalg.norm(xy[27] - xy[28]) / shoulder_width)
    # Nose is the available head reference; both wrists must be above it.
    result["wrists_above_head"] = float(max(xy[15, 1], xy[16, 1]) < xy[0, 1])
    torso = shoulders[:2] - hips[:2]
    result["torso_horizontal"] = float(abs(torso[0]) > abs(torso[1]))
    return result


class ExerciseTracker:
    """Motion counts include incorrect reps; correct_reps validates the whole cycle.

    Endpoint dwell is 120ms. Missing landmarks reset an incomplete cycle.
    Plank adds only adjacent valid intervals <= 0.5s, preventing gap credit.
    """
    def __init__(self, exercise):
        if exercise not in EXERCISES:
            raise ValueError(exercise)
        self.exercise = exercise
        self.reps = self.correct_reps = 0
        self.seconds = 0.0
        self.state = "ready"
        self.candidate = None
        self.candidate_since = 0.0
        self.last_time = None
        self.previous_valid = False
        self.cycle_min = 180.0
        self.cycle_ok = True
        self.score_sum = self.score_frames = 0
        self.history = deque(maxlen=18000)
        self.last_record = -math.inf
        self.started = None
        self.latest = {"feedback": "จัดกล้องให้เห็นร่างกายครบ", "valid": False, "score": 0,
                       "ai_label": "—", "confidence": 0, "metrics": {}}

    def invalidate(self, now, message="ไม่เห็นข้อต่อครบ กรุณาจัดตำแหน่งกล้องใหม่"):
        self.state, self.candidate = "ready", None
        self.previous_valid = False
        self.last_time = now
        self.latest = {"feedback": message, "valid": False, "score": 0,
                       "ai_label": "—", "confidence": 0, "metrics": {}}
        self._record(now, None)

    def _stable(self, endpoint, now):
        if endpoint != self.candidate:
            self.candidate, self.candidate_since = endpoint, now
        return endpoint is not None and now - self.candidate_since >= 0.12

    def update(self, f, now, ai_label=None, confidence=1.0, threshold=0.6):
        if self.last_time is not None and now - self.last_time > 0.5:
            self.invalidate(now, "สัญญาณภาพขาดช่วง")
        dt = 0 if self.last_time is None else max(0, now - self.last_time)
        self.last_time = now
        ex = self.exercise
        ai_match = ai_label is None or (ai_label.startswith(ex + "_") and confidence >= threshold)
        ai_ok = ai_match and (ai_label is None or not ai_label.endswith("wrong"))
        checks, tips = [], []
        endpoint = None
        if ex in ("squat", "pushup"):
            joint = f["knee" if ex == "squat" else "elbow"]
            endpoint = "up" if joint >= 160 else "down" if joint <= 110 else None
            if ex == "squat":
                checks = [f["back"] > 150, joint >= 70]
                if endpoint == "down":
                    checks.append(joint <= 100)
                tips = ["เหยียดหลังให้ตรง", "ย่อลึกเกินไป", "ย่อลงอีกเล็กน้อย"]
            else:
                checks = [f["body"] >= 160, bool(f["torso_horizontal"]), joint >= 65]
                if endpoint == "down":
                    checks.append(joint <= 100)
                tips = ["รักษาลำตัวให้ตรง", "จัดลำตัวขนานพื้น", "งอศอกลึกเกินไป", "ลดตัวลงอีกเล็กน้อย"]
        elif ex == "plank":
            checks = [160 <= f["body"] <= 180, bool(f["torso_horizontal"])]
            tips = ["ยกหรือลดสะโพกให้ลำตัวตรง", "จัดลำตัวขนานพื้น"]
        else:
            shoulders_up = min(f["shoulder_left"], f["shoulder_right"]) > 140
            closed = max(f["shoulder_left"], f["shoulder_right"]) < 35 and f["ankle_ratio"] < 1.15
            # A partial open can count as an incorrect attempt, but cannot be correct.
            opened = max(f["shoulder_left"], f["shoulder_right"]) > 65 or f["ankle_ratio"] > 1.4
            endpoint = "up" if closed else "down" if opened else None
            checks = [True] if closed else [shoulders_up, bool(f["wrists_above_head"]), f["ankle_ratio"] > 1.0]
            tips = ["ยกแขนให้สูงขึ้น", "ยกข้อมือให้เหนือศีรษะ", "กางขาให้กว้างขึ้น"]
        valid = all(checks) and ai_ok
        score = 100 * (sum(checks) + (int(ai_ok) if ai_label is not None else 0)) / (len(checks) + int(ai_label is not None))
        feedback = next((tip for check, tip in zip(checks, tips) if not check), "ท่าทางถูกต้อง")
        if not ai_match:
            feedback = "AI ยังไม่มั่นใจหรือพบคนละท่า กรุณาจัดมุมกล้อง"
        elif not ai_ok:
            feedback = "AI พบรูปแบบท่าที่ผิด: ปรับท่าทางตามตัวอย่างฝึก"
        if ex == "plank":
            if valid and self.previous_valid and dt <= 0.5:
                self.seconds += dt
            self.state = "holding" if valid else "paused"
        elif not ai_match:
            self.state, self.candidate = "ready", None
        else:
            # Track deepest point throughout descent/ascent, including brief overshoot.
            if self.state in ("up", "down"):
                if ex != "jj":
                    self.cycle_min = min(self.cycle_min, joint)
                    posture_ok = f["back"] > 150 if ex == "squat" else f["body"] >= 160 and bool(f["torso_horizontal"])
                    self.cycle_ok = self.cycle_ok and posture_ok and ai_ok
                elif self.state == "down" and endpoint == "down":
                    self.cycle_ok = self.cycle_ok or valid
            if self._stable(endpoint, now):
                if self.state == "ready" and endpoint == "up":
                    self.state, self.cycle_min, self.cycle_ok = "up", 180.0, True
                elif self.state == "up" and endpoint == "down":
                    self.state = "down"
                    if ex == "jj":
                        self.cycle_ok = valid
                elif self.state == "down" and endpoint == "up":
                    self.reps += 1
                    depth_ok = True if ex == "jj" else (70 if ex == "squat" else 65) <= self.cycle_min <= 100
                    self.correct_reps += int(self.cycle_ok and depth_ok)
                    self.state, self.cycle_min, self.cycle_ok = "up", 180.0, True
        self.previous_valid = valid
        self.score_sum += score
        self.score_frames += 1
        self.latest = {"feedback": feedback, "valid": valid, "score": score,
                       "ai_label": ai_label or "Rule-only", "confidence": confidence, "metrics": {k: f[k] for k in METRIC_COLUMNS}}
        self._record(now, score)

    def _record(self, now, score):
        if self.started is None:
            self.started = now
        if now - self.last_record >= 0.1:
            self.history.append({"elapsed_s": round(now - self.started, 2), "form_score": score,
                                 "reps": self.reps, "correct_reps": self.correct_reps, "plank_s": self.seconds,
                                 "knee": self.latest["metrics"].get("knee"), "elbow": self.latest["metrics"].get("elbow"),
                                 "body": self.latest["metrics"].get("body")})
            self.last_record = now

    def snapshot(self):
        return {**self.latest, "exercise": self.exercise, "reps": self.reps, "correct_reps": self.correct_reps,
                "seconds": self.seconds, "state": self.state, "frames": self.score_frames,
                "average_score": self.score_sum / max(1, self.score_frames), "history": list(self.history)}
