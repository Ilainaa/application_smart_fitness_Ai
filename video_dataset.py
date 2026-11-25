"""Persistent video annotations and deterministic, non-appending CSV export."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

import av
import cv2
import numpy as np

from fitness_core import CLASSES, CSV_COLUMNS, RAW_COLUMNS, SCHEMA_VERSION, LandmarkEMA, extract_features, visible

ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT = ROOT / "dataset/video_project"
DEFAULT_OUTPUT = ROOT / "dataset/video_pose_data.csv"


@contextmanager
def project_lock(project):
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    lock = project / ".write.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError("โปรเจกต์กำลังบันทึกหรือส่งออกอยู่ กรุณารอให้เสร็จ") from None
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def load_project(project=DEFAULT_PROJECT):
    path = Path(project) / "annotations.json"
    if not path.exists():
        return {"version": 1, "revision": 0, "clips": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("clips"), list):
        raise ValueError("ไม่รองรับรูปแบบไฟล์ annotations.json นี้")
    return data


def _save_project(project, data):
    data["revision"] += 1
    atomic_json(Path(project) / "annotations.json", data)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def clip_path(project, clip):
    base = Path(project).resolve()
    path = (base / clip["file"]).resolve()
    if not path.is_relative_to(base / "videos") or not path.is_file():
        raise ValueError("ไม่พบไฟล์วิดีโอในโฟลเดอร์ videos ของโปรเจกต์")
    return path


def video_frames(path, extra_rotation=0):
    """Use presentation timestamps (including VFR phones), normalized to first frame.

    PyAV rotation is counterclockwise. The manual adjustment is clockwise.
    Yield upright BGR pixels, not browser-dependent rotation metadata.
    """
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("ไฟล์ไม่มี video stream")
        origin, previous = None, -1.0
        for frame in container.decode(video=0):
            if frame.time is None:
                raise ValueError("วิดีโอไม่มี timestamp: กรุณาแปลงเป็น MP4 H.264 ก่อน")
            if origin is None:
                origin = frame.time
            seconds = float(frame.time - origin)
            if seconds <= previous:
                continue
            previous = seconds
            image = frame.to_ndarray(format="bgr24")
            rotation = int(getattr(frame, "rotation", 0))
            image = np.ascontiguousarray(np.rot90(image, round(rotation / 90) - extra_rotation // 90))
            if image.shape[1] > 960:
                image = cv2.resize(image, (960, round(image.shape[0] * 960 / image.shape[1])))
            yield seconds, image


def inspect_video(path):
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("ไฟล์ไม่มี video stream")
        stream = container.streams.video[0]
        frame = next(container.decode(video=0), None)
        if frame is None or frame.time is None:
            raise ValueError("อ่านเฟรมและ timestamp ของวิดีโอไม่ได้")
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        width, height = frame.width, frame.height
    if not math.isfinite(duration) or duration <= 0:
        # Some containers omit duration; a sequential scan still uses actual timestamps.
        duration = 0.0
        for seconds, _ in video_frames(path):
            duration = seconds + 1 / max(1, fps)
    if duration <= 0:
        raise ValueError("วิดีโอว่าง")
    return {"duration": duration, "fps": fps, "width": width, "height": height}


def preview_frame(path, seconds, rotation=0):
    last = None
    for timestamp, frame in video_frames(path, rotation):
        last = (timestamp, frame)
        if timestamp >= seconds - 1e-6:
            return last
    if last is None:
        raise ValueError("วิดีโอไม่มีเฟรมที่อ่านได้")
    return last


def _identity(subject, recording):
    if not isinstance(subject, str) or not isinstance(recording, str) or not subject.strip() or not recording.strip():
        raise ValueError("กรุณากรอก Subject ID และรอบการถ่าย")
    return subject.strip(), recording.strip()


def import_video(source, name, subject, recording, project=DEFAULT_PROJECT):
    """Copy a binary stream; SHA256 identity prevents renamed duplicate imports."""
    subject, recording = _identity(subject, recording)
    project = Path(project)
    suffix = Path(name).suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}:
        raise ValueError("รองรับ MP4, MOV, M4V, AVI, MKV และ WebM")
    videos = project / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=videos, suffix=suffix)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as dest:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                dest.write(chunk)
        clip_id = digest.hexdigest()
        info = inspect_video(temp)
        with project_lock(project):
            data = load_project(project)
            duplicate = next((c for c in data["clips"] if c["id"] == clip_id), None)
            if duplicate:
                return duplicate, False
            destination = videos / (clip_id + suffix)
            os.replace(temp, destination)
            clip = {"id": clip_id, "name": Path(name).name, "file": f"videos/{destination.name}",
                    "subject_id": subject, "recording": recording, "rotation": 0, "segments": [], **info}
            data["clips"].append(clip)
            _save_project(project, data)
            return clip, True
    finally:
        Path(temp).unlink(missing_ok=True)


def validate_segments(segments, duration):
    clean = []
    for segment in segments:
        start, end = float(segment["start"]), float(segment["end"])
        label = segment["label"]
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration + 1e-6:
            raise ValueError("ช่วงเวลาต้องอยู่ในคลิป และเวลาเริ่มต้องน้อยกว่าเวลาจบ")
        if label not in CLASSES:
            raise ValueError(f"ไม่รู้จักคลาส: {label}")
        clean.append({"start": start, "end": end, "label": label})
    clean.sort(key=lambda s: s["start"])
    if any(b["start"] < a["end"] - 1e-9 for a, b in zip(clean, clean[1:])):
        raise ValueError("ช่วงเวลาทับกัน กรุณาแก้ไขก่อนบันทึก")
    return clean


def update_clip(project, clip_id, revision, *, segments=None, subject=None, recording=None, rotation=None, annotation_mode=None):
    with project_lock(project):
        data = load_project(project)
        if data["revision"] != revision:
            raise ValueError("ข้อมูลเปลี่ยนจากอีกหน้าต่างแล้ว กรุณารีเฟรชก่อนแก้ไข")
        clip = next(c for c in data["clips"] if c["id"] == clip_id)
        if annotation_mode is not None:
            if annotation_mode not in ("segments", "frames") or (annotation_mode == "frames" and not clip.get("frame_set")):
                raise ValueError("กรุณาแตกภาพก่อนเลือกส่งออกรายภาพ")
            clip["annotation_mode"] = annotation_mode
        if segments is not None:
            clip["segments"] = validate_segments(segments, clip["duration"])
        if subject is not None or recording is not None:
            clip["subject_id"], clip["recording"] = _identity(subject, recording)
        if rotation is not None:
            if rotation not in (0, 90, 180, 270):
                raise ValueError("มุมหมุนต้องเป็น 0, 90, 180 หรือ 270")
            clip["rotation"] = rotation
        _save_project(project, data)


def recording_id(clip):
    key = clip["subject_id"] + "\0" + clip["recording"]
    return "video_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def has_annotations(clip):
    if clip.get("annotation_mode", "segments") == "frames":
        return any(f["label"] in CLASSES for f in clip.get("frame_set", {}).get("frames", []))
    return bool(clip["segments"])


def export_dataset(project=DEFAULT_PROJECT, output=DEFAULT_OUTPUT, sample_fps=5.0, progress=None, pose_factory=None):
    """Regenerate the full annotated dataset atomically. Never append duplicates.

    Every labelled frame feeds Pose/EMA; only CSV sampling is limited to sample_fps.
    Unlabelled transitions never become training samples. An empty/failed export
    leaves the previous CSV intact. The original live capture CSV is independent.
    """
    if not math.isfinite(sample_fps) or not 0 < sample_fps <= 30:
        raise ValueError("sample_fps ต้องมากกว่า 0 และไม่เกิน 30")
    if pose_factory is None:
        from camera_pipeline import create_pose
        pose_factory = create_pose
    from camera_pipeline import landmarks_array
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with project_lock(project):
        data = load_project(project)
        clips = [c for c in data["clips"] if has_annotations(c)]
        if not clips:
            raise ValueError("ยังไม่มีช่วงเวลาที่ติดป้ายกำกับ")
        fd, temp = tempfile.mkstemp(dir=output.parent, suffix=".csv.tmp")
        report = {"export_id": uuid.uuid4().hex, "revision": data["revision"], "sample_fps": sample_fps,
                  "rows": 0, "classes": {}, "clips": []}
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as dest:
                writer = csv.DictWriter(dest, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                for clip_index, clip in enumerate(clips):
                    path = clip_path(project, clip)
                    if file_sha256(path) != clip["id"]:
                        raise ValueError(f"ไฟล์ต้นฉบับเปลี่ยนไป: {clip['name']} กรุณานำเข้าใหม่")
                    if clip.get("annotation_mode", "segments") == "frames":
                        from frame_labeling import export_frame_rows
                        rows = export_frame_rows(clip)
                        writer.writerows(rows)
                        for row in rows:
                            report["classes"][row["label"]] = report["classes"].get(row["label"], 0) + 1
                        report["rows"] += len(rows)
                        report["clips"].append({"id": clip["id"], "name": clip["name"],
                            "session_id": recording_id(clip), "mode": "frames",
                            "segments": [{"label": label, "saved": sum(r["label"] == label for r in rows), "skipped": 0}
                                         for label in sorted({r["label"] for r in rows})]})
                        if progress:
                            progress((clip_index + 1) / len(clips), clip["name"])
                        continue
                    segments = validate_segments(clip["segments"], clip["duration"])
                    stats = {"id": clip["id"], "name": clip["name"], "subject_id": clip["subject_id"],
                             "recording": clip["recording"], "session_id": recording_id(clip),
                             "segments": [{**s, "saved": 0, "skipped": 0} for s in segments]}
                    smoother, segment_index, current = LandmarkEMA(), 0, None
                    next_sample, last_time = 0.0, None
                    with pose_factory() as pose:
                        for frame_index, (seconds, frame) in enumerate(video_frames(path, clip.get("rotation", 0))):
                            while segment_index < len(segments) and seconds >= segments[segment_index]["end"]:
                                segment_index += 1
                            if segment_index == len(segments):
                                break
                            segment = segments[segment_index]
                            if seconds < segment["start"]:
                                continue
                            if current != segment_index or (last_time is not None and seconds - last_time > 0.5):
                                smoother.reset()
                                next_sample = seconds
                            current, last_time = segment_index, seconds
                            due = seconds + 1e-8 >= next_sample
                            if due:
                                next_sample = seconds + 1 / sample_fps
                            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            features, raw = None, None
                            if result.pose_landmarks:
                                raw = landmarks_array(result.pose_landmarks)
                                if visible(raw, segment["label"].split("_")[0]):
                                    try:
                                        features = extract_features(smoother.update(raw), frame.shape[1], frame.shape[0])
                                    except ValueError:
                                        smoother.reset()
                                else:
                                    smoother.reset()
                            else:
                                smoother.reset()
                            if due and features is not None:
                                label = segment["label"]
                                writer.writerow({"schema_version": SCHEMA_VERSION, "session_id": recording_id(clip),
                                                 "subject_id": clip["subject_id"], "label": label,
                                                 "timestamp": f"video:{clip['id']}:{seconds:.6f}s",
                                                 **dict(zip(RAW_COLUMNS, raw.flatten())), **features})
                                stats["segments"][segment_index]["saved"] += 1
                                report["rows"] += 1
                                report["classes"][label] = report["classes"].get(label, 0) + 1
                            elif due:
                                stats["segments"][segment_index]["skipped"] += 1
                            if progress and frame_index % 30 == 0:
                                progress((clip_index + seconds / clip["duration"]) / len(clips), clip["name"])
                    report["clips"].append(stats)
                    if progress:
                        progress((clip_index + 1) / len(clips), clip["name"])
            if not report["rows"]:
                raise ValueError("ไม่พบเฟรมที่เห็นข้อต่อครบในช่วงที่เลือก ไม่มีการเขียนทับ CSV เดิม")
            # All decoding/inference succeeded before publishing a replacement.
            os.replace(temp, output)
            atomic_json(output.with_suffix(".report.json"), report)
            return report
        finally:
            Path(temp).unlink(missing_ok=True)
