"""Collect real, explicitly labelled poses. Run --help for camera options."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid
import cv2
from camera_pipeline import camera_source, open_capture, create_pose, landmarks_array, draw_skeleton, ROOT
from fitness_core import (CLASS_KEYS, CSV_COLUMNS, RAW_COLUMNS, SCHEMA_VERSION,
                          LandmarkEMA, extract_features, visible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index or DroidCam/IP camera URL")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset/pose_data.csv")
    parser.add_argument("--subject", required=True, help="Anonymous participant ID, e.g. person01")
    parser.add_argument("--sample-fps", type=float, default=5)
    args = parser.parse_args()
    if not 0 < args.sample_fps <= 30:
        parser.error("--sample-fps must be in (0, 30]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = args.output.exists() and args.output.stat().st_size > 0
    if existing:
        with args.output.open(newline="", encoding="utf-8") as f:
            if next(csv.reader(f)) != CSV_COLUMNS:
                parser.error("CSV schema differs. Use a new --output file.")
    session = uuid.uuid4().hex
    print("Click the camera window, then select a class:")
    for key, label in CLASS_KEYS.items():
        print(f"  {key}: {label}")
    print("SPACE: record/pause | n: new recording group | q/ESC: exit")
    print("Recording starts PAUSED; select a label, settle into pose, then press SPACE.")
    cap = open_capture(camera_source(args.source))
    smoother = LandmarkEMA()
    label, recording, count, last_sample = None, False, 0, 0.0
    try:
        with create_pose() as pose, args.output.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not existing:
                writer.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera stopped sending frames")
                if frame.shape[1] > 960:
                    frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                now, features, raw = time.monotonic(), None, None
                if result.pose_landmarks and label:
                    raw = landmarks_array(result.pose_landmarks)
                    if visible(raw, label.split("_")[0]):
                        points = smoother.update(raw)
                        try:
                            features = extract_features(points, frame.shape[1], frame.shape[0])
                        except ValueError:
                            pass
                        draw_skeleton(frame, points, features is not None)
                    else:
                        smoother.reset()
                else:
                    smoother.reset()
                if recording and features is not None and now - last_sample >= 1 / args.sample_fps:
                    writer.writerow({"schema_version": SCHEMA_VERSION, "session_id": session,
                                     "subject_id": args.subject, "timestamp": datetime.now(timezone.utc).isoformat(),
                                     "label": label, **dict(zip(RAW_COLUMNS, raw.flatten())), **features})
                    f.flush()
                    count, last_sample = count + 1, now
                status = "REC" if recording else "PAUSED"
                cv2.putText(frame, f"{status} | {label or 'SELECT A CLASS'} | saved: {count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 240, 180), 2)
                cv2.putText(frame, "SPACE record/pause | n new group | q quit", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if features is None:
                    cv2.putText(frame, "No usable pose / select class", (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 255), 2)
                cv2.imshow("Smart Fitness AI - Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty("Smart Fitness AI - Data Collection", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if chr(key) in CLASS_KEYS:
                    label, recording = CLASS_KEYS[chr(key)], False
                    smoother.reset()
                elif key == ord(" ") and label:
                    recording = not recording
                elif key == ord("n"):
                    session, recording = uuid.uuid4().hex, False
                    smoother.reset()
                    print("New recording group:", session)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Saved {count} new samples to {args.output}")


if __name__ == "__main__":
    main()
