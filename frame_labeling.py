"""Sample videos for human review; keep the features shown in the review gallery."""
from pathlib import Path
import math
import uuid

import cv2
import numpy as np

from fitness_core import CLASSES, EXERCISES, FEATURE_COLUMNS, RAW_COLUMNS, SCHEMA_VERSION, LandmarkEMA, extract_features, visible
from video_dataset import (load_project, project_lock, _save_project, clip_path,
                           file_sha256, video_frames, recording_id)


def generate_frames(project, clip_id, revision, exercise, interval=1.0, pose_factory=None, progress=None):
    from camera_pipeline import create_pose, landmarks_array
    if exercise not in EXERCISES or not math.isfinite(interval) or not .1 <= interval <= 10:
        raise ValueError("เลือกท่าและช่วงห่างภาพระหว่าง 0.1–10 วินาที")
    with project_lock(project):
        data = load_project(project)
        if data["revision"] != revision:
            raise ValueError("ข้อมูลเปลี่ยนแล้ว กรุณารีเฟรช")
        clip = next(c for c in data["clips"] if c["id"] == clip_id)
        source = clip_path(project, clip)
        if file_sha256(source) != clip_id:
            raise ValueError("ไฟล์ต้นฉบับเปลี่ยนไป กรุณานำเข้าใหม่")
        # shortcut: old image folders remain for recovery; add cleanup if disk use becomes significant.
        folder = Path(project) / "frames" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        frames, smoother, next_sample, last_time = [], LandmarkEMA(), 0.0, None
        with (pose_factory or create_pose)() as pose:
            for index, (seconds, image) in enumerate(video_frames(source, clip.get("rotation", 0))):
                if last_time is not None and seconds - last_time > .5:
                    smoother.reset()
                last_time = seconds
                result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                raw = points = features = None
                if result.pose_landmarks:
                    raw = landmarks_array(result.pose_landmarks)
                    points = raw
                    if visible(raw, exercise):
                        points = smoother.update(raw)
                        try:
                            features = extract_features(points, image.shape[1], image.shape[0])
                        except ValueError:
                            smoother.reset()
                    else:
                        smoother.reset()
                else:
                    smoother.reset()
                if seconds + 1e-8 >= next_sample:
                    image_path = folder / f"{index}.jpg"
                    if not cv2.imwrite(str(image_path), image):
                        raise OSError("บันทึกภาพไม่ได้")
                    frames.append({"index": index, "time": seconds,
                                   "image": image_path.relative_to(Path(project)).as_posix(),
                                   "points": points.tolist() if points is not None else None,
                                   "raw": raw.flatten().tolist() if raw is not None else None,
                                   "features": features, "label": None})
                    next_sample = (math.floor((seconds + 1e-8) / interval) + 1) * interval
                if progress and index % 30 == 0:
                    progress(min(1.0, seconds / clip["duration"]))
        if not frames:
            raise ValueError("ไม่พบเฟรมในวิดีโอ")
        clip["frame_set"] = {"exercise": exercise, "interval": interval,
                             "rotation": clip.get("rotation", 0), "frames": frames}
        clip["annotation_mode"] = "frames"
        _save_project(project, data)
        return len(frames)


def label_frames(project, clip_id, revision, indices, label):
    with project_lock(project):
        data = load_project(project)
        if data["revision"] != revision:
            raise ValueError("ข้อมูลเปลี่ยนแล้ว กรุณารีเฟรช")
        clip = next(c for c in data["clips"] if c["id"] == clip_id)
        frame_set = clip["frame_set"]
        allowed = {c for c in CLASSES if c.startswith(frame_set["exercise"] + "_")}
        if label not in allowed | {None, "skip"}:
            raise ValueError("คลาสไม่ตรงกับท่าของชุดภาพ")
        if frame_set["rotation"] != clip.get("rotation", 0):
            raise ValueError("แนวภาพเปลี่ยนแล้ว กรุณาแตกภาพใหม่ก่อนติดป้าย")
        selected = [f for f in frame_set["frames"] if f["index"] in indices]
        if not selected or {f["index"] for f in selected} != set(indices):
            raise ValueError("กรุณาเลือกภาพที่มีอยู่ในชุดนี้")
        if label in allowed and any(f["features"] is None for f in selected):
            raise ValueError("บางภาพไม่มีข้อต่อครบ ให้ข้ามภาพนั้นหรือเลือกเฉพาะภาพที่ใช้ได้")
        for frame in selected:
            frame["label"] = label
        _save_project(project, data)


def review_image(project, frame, overlay=True):
    from camera_pipeline import draw_skeleton
    base = Path(project).resolve()
    path = (base / frame["image"]).resolve()
    if not path.is_relative_to(base / "frames"):
        raise ValueError("พาธภาพไม่ถูกต้อง")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("ไม่พบภาพ กรุณาแตกภาพใหม่")
    if overlay and frame["points"] is not None:
        points = np.asarray(frame["points"])
        draw_skeleton(image, points, True)
        h, w = image.shape[:2]
        for i, p in enumerate(points):
            if p[3] >= .55 and np.all((p[:2] >= 0) & (p[:2] <= 1)):
                cv2.putText(image, str(i), (int(p[0] * (w - 1)), int(p[1] * (h - 1))),
                            cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def export_frame_rows(clip):
    frame_set = clip["frame_set"]
    if frame_set["rotation"] != clip.get("rotation", 0):
        raise ValueError("แนวภาพเปลี่ยนแล้ว กรุณาแตกภาพและตรวจป้ายใหม่")
    rows, seen = [], set()
    for frame in frame_set["frames"]:
        label = frame["label"]
        if label in (None, "skip"):
            continue
        if label not in CLASSES or not label.startswith(frame_set["exercise"] + "_"):
            raise ValueError("คลาสของภาพไม่ถูกต้อง")
        timestamp = float(frame["time"])
        if not math.isfinite(timestamp) or not 0 <= timestamp < clip["duration"] or timestamp in seen:
            raise ValueError("เวลาของภาพไม่ถูกต้องหรือซ้ำกัน")
        seen.add(timestamp)
        features, raw = frame["features"], np.asarray(frame["raw"], dtype=float)
        if (features is None or set(features) != set(FEATURE_COLUMNS) or raw.shape != (132,)
                or not np.isfinite(raw).all() or not np.isfinite(list(features.values())).all()
                or not visible(raw.reshape(33, 4), frame_set["exercise"])):
            raise ValueError("ข้อมูลข้อต่อของภาพไม่สมบูรณ์ กรุณาข้ามภาพหรือแตกภาพใหม่")
        rows.append({"schema_version": SCHEMA_VERSION, "session_id": recording_id(clip),
                     "subject_id": clip["subject_id"], "label": label,
                     "timestamp": f"video:{clip['id']}:{timestamp:.6f}s",
                     **dict(zip(RAW_COLUMNS, raw)), **features})
    return rows
