"""Offline video labelling page. Open through: streamlit run app.py."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from fitness_core import CLASSES, EXERCISES
from video_dataset import (DEFAULT_PROJECT, DEFAULT_OUTPUT, load_project, import_video,
                           update_clip, clip_path, preview_frame, export_dataset, has_annotations)
from frame_labeling_ui import show_frame_labeler, label_name
from app import render_shell

st.set_page_config(page_title="Video Labeler · Smart Fitness AI", page_icon=":material/video_library:", layout="wide", initial_sidebar_state="collapsed")
render_shell("labeler")
st.markdown("""<style>
h1 {font-size:2rem !important;letter-spacing:-.025em;}
h2,h3 {letter-spacing:-.015em;}
[data-testid="stCaptionContainer"] {color:#b8ceca;line-height:1.6;}
[data-testid="stMetric"] {background:#123238;border:1px solid #2b4c4c;border-radius:12px;padding:14px;}
[data-testid="stMetricValue"] {font-size:1.6rem;}
[data-testid="stForm"] {border-color:#2b4c4c;border-radius:14px;padding:20px;}
[data-testid="stForm"] [data-testid="stImage"] img {max-height:260px;object-fit:contain;}
button {min-height:48px;}
[data-baseweb="tab"] {font-size:16px;padding:12px 20px;}
@media(max-width:640px) {[data-testid="stForm"]{padding:12px;}}
</style>""", unsafe_allow_html=True)
st.caption("SMART FITNESS AI / DATA STUDIO")
st.title("เตรียมข้อมูลเทรน AI")
st.caption("ตรวจภาพ ติดป้ายให้ถูกต้อง แล้วสร้างชุดข้อมูลสำหรับโมเดลของคุณ")
with st.expander("ข้อมูลและป้ายกำกับบันทึกบนคอมเครื่องนี้ ปิดเว็บแล้วกลับมาทำต่อได้"):
    st.caption("ไฟล์งาน: dataset/video_project/annotations.json\n\nไฟล์ฝึก: dataset/video_pose_data.csv")

# Stop a live session before switching to offline work; do not leave a camera running.
if st.session_state.get("active"):
    st.warning("กรุณากลับหน้าฝึกซ้อมและจบเซสชันก่อนนำเข้าวิดีโอ")
    st.page_link("app.py", label="กลับหน้าฝึกซ้อม")
    st.stop()

notice = st.session_state.pop("video_notice", None)
if notice:
    st.success(notice)

try:
    project = load_project(DEFAULT_PROJECT)
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

label_tab, import_tab, export_tab = st.tabs(["ตรวจและติดป้าย", "เพิ่มคลิป", "ส่งออกและเทรน"])

with import_tab:
    with st.expander("1. เพิ่มคลิปใหม่", expanded=True):
        with st.form("import_video", clear_on_submit=False):
            uploaded = st.file_uploader("เลือกคลิปจากมือถือหรือคอม (เลือกได้หลายไฟล์)",
                                        type=["mp4", "mov", "m4v", "avi", "mkv", "webm"], accept_multiple_files=True)
            local_path = st.text_input("หรือระบุพาธคลิปบนคอมเซิร์ฟเวอร์ (สำหรับไฟล์ใหญ่)", placeholder=r"C:\Videos\squat.mp4")
            cols = st.columns(2)
            subject = cols[0].text_input("Subject ID", "person01")
            recording = cols[1].text_input("รอบการถ่าย", "take01")
            st.caption("คลิปที่ถ่ายในรอบเดียวกันใช้ Subject ID และรอบเดียวกัน เช่น take01; "
                       "เปลี่ยนเป็น take02 เมื่อถ่ายรอบใหม่จริง อย่าแยกเฟรมติดกันเป็นคนละรอบ")
            import_pressed = st.form_submit_button("นำเข้าและบันทึกคลิป", type="primary")
        if import_pressed:
            try:
                if bool(uploaded) == bool(local_path.strip()):
                    raise ValueError("เลือกไฟล์อัปโหลดหรือพาธในเครื่องอย่างใดอย่างหนึ่ง")
                new_count = duplicate_count = 0
                with st.spinner("กำลังตรวจและบันทึกคลิป…"):
                    if uploaded:
                        for source in uploaded:
                            source.seek(0)
                            _, added = import_video(source, source.name, subject, recording, DEFAULT_PROJECT)
                            new_count += int(added)
                            duplicate_count += int(not added)
                    else:
                        path = Path(local_path.strip().strip('"'))
                        with path.open("rb") as source:
                            _, added = import_video(source, path.name, subject, recording, DEFAULT_PROJECT)
                        new_count, duplicate_count = int(added), int(not added)
                st.session_state.video_notice = f"นำเข้าใหม่ {new_count} คลิป · ข้ามไฟล์ซ้ำ {duplicate_count} คลิป"
                st.rerun()
            except Exception as exc:
                st.error(f"นำเข้าไม่สำเร็จ: {exc} (คลิปที่นำเข้าสำเร็จก่อนหน้านี้ยังอยู่)")

if not project["clips"]:
    label_tab.info("ยังไม่มีคลิป เริ่มที่แท็บ «เพิ่มคลิป» แล้วกลับมาติดป้ายที่นี่")
    st.stop()

with label_tab:
    st.subheader("เลือกคลิปที่ต้องการตรวจ")
    by_id = {c["id"]: c for c in project["clips"]}
    selected = st.selectbox("คลิปที่ต้องการทำงาน", list(by_id), format_func=lambda k: (
        f"{by_id[k]['name']} · {by_id[k]['subject_id']} / {by_id[k]['recording']} · {k[:6]}"))
    clip = by_id[selected]
    try:
        path = clip_path(DEFAULT_PROJECT, clip)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.caption(f"ความยาว {clip['duration']:.2f} วินาที · {clip['fps']:.2f} FPS เฉลี่ย · "
               f"ติดป้ายแล้ว {len(clip['segments'])} ช่วง")

    with st.expander("แก้ไขผู้ถ่าย รอบการถ่าย และแนวภาพ"):
        with st.form(f"identity_{selected}"):
            edit_subject = st.text_input("Subject ID ของคลิป", clip["subject_id"])
            edit_recording = st.text_input("รอบการถ่ายของคลิป", clip["recording"])
            rotation = st.selectbox("หมุนเพิ่มตามเข็มนาฬิกา (ภาพตัวอย่างด้านล่างและข้อมูลฝึก)",
                                    [0, 90, 180, 270], index=[0, 90, 180, 270].index(clip["rotation"]))
            st.caption("ระบบอ่านแนวภาพจาก metadata ให้อัตโนมัติแล้ว ใช้ตัวเลือกนี้เฉพาะเมื่อภาพตัวอย่างยังเอียง")
            if st.form_submit_button("บันทึกข้อมูลคลิป"):
                try:
                    update_clip(DEFAULT_PROJECT, selected, project["revision"], subject=edit_subject,
                                recording=edit_recording, rotation=rotation)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    with st.expander("ดูวิดีโอต้นฉบับ / ตรวจเฟรมตามเวลา"):
        left, right = st.columns(2)
        with left:
            st.video(str(path))
            st.caption("ใช้วิดีโอเพื่อดูการเคลื่อนไหว หาก MOV/HEVC เล่นในเบราว์เซอร์ไม่ได้ "
                       "ยังดูเฟรมด้วยปุ่มด้านขวาได้ หรือแปลงคลิปเป็น MP4 H.264")
        with right:
            preview_time = st.number_input("ดูเฟรมที่วินาที", min_value=0.0, max_value=float(clip["duration"]),
                                           value=0.0, step=0.1, format="%.3f", key=f"preview_t_{selected}")
            if st.button("แสดงเฟรม ณ เวลาที่เลือก"):
                try:
                    t, frame = preview_frame(path, preview_time, clip["rotation"])
                    st.session_state.video_preview = {"id": selected, "rotation": clip["rotation"], "time": t, "frame": frame}
                except Exception as exc:
                    st.error(f"อ่านเฟรมไม่ได้: {exc}")
            preview = st.session_state.get("video_preview")
            if preview and preview["id"] == selected and preview["rotation"] == clip["rotation"]:
                st.image(preview["frame"], channels="BGR", caption=f"เฟรมจริง {preview['time']:.3f} วินาที", use_container_width=True)

    mode = st.radio("วิธีติดป้ายและส่งออกของคลิปนี้", ["segments", "frames"],
                    format_func=lambda v: "เลือกช่วงเวลา" if v == "segments" else "ติดป้ายรายภาพ",
                    index=0 if clip.get("annotation_mode", "segments") == "segments" else 1,
                    key=f"annotation_mode_{selected}", horizontal=True)
    if mode != clip.get("annotation_mode", "segments") and (mode == "segments" or clip.get("frame_set")):
        try:
            update_clip(DEFAULT_PROJECT, selected, project["revision"], annotation_mode=mode)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    st.caption("ส่งออกเพียงโหมดที่เลือกต่อคลิป ป้ายอีกโหมดยังเก็บไว้และกลับมาใช้ได้")
    if mode == "frames":
        show_frame_labeler(DEFAULT_PROJECT, project, clip)
    else:
        exercise = st.selectbox("กรองคลาสตามท่า", list(EXERCISES), format_func=EXERCISES.get)
        with st.form(f"segment_{selected}"):
            a, b, c = st.columns(3)
            start = a.number_input("เริ่ม (วินาที)", min_value=0.0, max_value=float(clip["duration"]), value=0.0, step=0.1, format="%.3f")
            end = b.number_input("จบ (วินาที ไม่รวมเฟรมตรงจุดจบ)", min_value=0.0, max_value=float(clip["duration"]),
                                 value=float(clip["duration"]), step=0.1, format="%.3f")
            label = c.selectbox("ป้ายกำกับ", [name for name in CLASSES if name.startswith(exercise + "_")], format_func=label_name)
            st.caption("เลือกเฉพาะช่วงที่อยู่ในสถานะนั้นจริง เช่น ยืน หรือย่อลงสุด "
                       "เว้นช่วงเปลี่ยนท่าไว้ ไม่ควรติดป้าย squat_up ให้ทั้งคลิปที่มีการย่อตัวด้วย")
            if st.form_submit_button("เพิ่มช่วงและบันทึกป้ายกำกับ", type="primary"):
                try:
                    update_clip(DEFAULT_PROJECT, selected, project["revision"],
                                segments=clip["segments"] + [{"start": start, "end": end, "label": label}])
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

        if clip["segments"]:
            st.dataframe(pd.DataFrame(clip["segments"]), hide_index=True, use_container_width=True)
            with st.expander("แก้ไขหรือลบช่วงที่บันทึกแล้ว"):
                index = st.selectbox("ช่วงที่จะแก้ไข", range(len(clip["segments"])),
                                      format_func=lambda i: f"{i + 1}: {clip['segments'][i]['start']:.3f}–{clip['segments'][i]['end']:.3f}s {clip['segments'][i]['label']}")
                segment = clip["segments"][index]
                with st.form(f"edit_segment_{selected}_{index}_{project['revision']}"):
                    edit_start = st.number_input("เวลาเริ่มใหม่", value=segment["start"], step=0.1, format="%.3f")
                    edit_end = st.number_input("เวลาจบใหม่", value=segment["end"], step=0.1, format="%.3f")
                    edit_label = st.selectbox("คลาสใหม่", CLASSES, index=CLASSES.index(segment["label"]))
                    save = st.form_submit_button("บันทึกการแก้ไข")
                    delete = st.form_submit_button("ลบช่วงนี้")
                if save or delete:
                    segments = [dict(s) for s in clip["segments"]]
                    if delete:
                        segments.pop(index)
                    else:
                        segments[index] = {"start": edit_start, "end": edit_end, "label": edit_label}
                    try:
                        update_clip(DEFAULT_PROJECT, selected, project["revision"], segments=segments)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

with export_tab:
    st.subheader("ตรวจความพร้อมของชุดข้อมูล")
    summary = []
    for name in CLASSES:
        matches = [(c, s) for c in project["clips"] for s in c["segments"] if s["label"] == name and c.get("annotation_mode", "segments") == "segments"]
        images = [(c, f) for c in project["clips"] if c.get("annotation_mode") == "frames"
                  for f in c.get("frame_set", {}).get("frames", []) if f["label"] == name]
        summary.append({"ภาพ": len(images), "คลาส": name, "ช่วง": len(matches), "วินาที": round(sum(s["end"] - s["start"] for _, s in matches), 2),
                        "รอบอิสระ": len({(c["subject_id"], c["recording"]) for c, _ in matches + images})})
    st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
    st.caption("แต่ละคลาสควรมีข้อมูลจากหลายรอบจริง อย่างน้อย 2 รอบ "
               "และแนะนำ 4 รอบขึ้นไป เพื่อให้แบ่ง Train/Test ได้")
    st.caption("อัตราส่งออกต่อวินาทีใช้เฉพาะช่วงเวลา; รายภาพส่งออกเฉพาะภาพที่ยืนยันป้ายแล้ว")
    sample_fps = st.number_input("จำนวนตัวอย่างสูงสุดต่อวินาที", min_value=1, max_value=30, value=5)
    st.info("ส่งออกใหม่จะสร้าง video_pose_data.csv จากป้ายกำกับปัจจุบันทั้งหมดแทนไฟล์ส่งออกเดิม "
            "ไม่ต่อแถวซ้ำ และไม่แก้ pose_data.csv ที่เก็บจากกล้องสด")
    if st.button("แปลงคลิปทั้งหมดเป็น CSV", type="primary", disabled=not any(has_annotations(c) for c in project["clips"])):
        bar = st.progress(0.0)
        try:
            with st.spinner("กำลังตรวจจับข้อต่อและสร้างชุดข้อมูล กรุณารอให้เสร็จก่อนปิดหน้าเว็บ…"):
                report = export_dataset(DEFAULT_PROJECT, DEFAULT_OUTPUT, sample_fps=float(sample_fps),
                                        progress=lambda fraction, name: bar.progress(min(1.0, fraction), text=name))
            st.session_state.video_export = report
        except Exception as exc:
            st.error(f"ส่งออกไม่สำเร็จ: {exc}")

    try:
        report = json.loads(DEFAULT_OUTPUT.with_suffix(".report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = None
    if report:
        if report["revision"] != project["revision"]:
            st.warning("ป้ายกำกับเปลี่ยนหลังส่งออกครั้งล่าสุด กรุณาสร้าง CSV ใหม่ก่อนเทรน")
        st.success(f"ผลส่งออกครั้งล่าสุด: {report['rows']:,} แถว")
        details = [{"คลิป": c["name"], **s} for c in report["clips"] for s in c["segments"]]
        st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)
        if any(s["saved"] == 0 for c in report["clips"] for s in c["segments"]):
            st.warning("บางช่วงไม่มีตัวอย่างที่ใช้ได้ ตรวจแนวภาพ แสง และว่ามองเห็นข้อต่อครบหรือไม่")
    if DEFAULT_OUTPUT.exists():
        with DEFAULT_OUTPUT.open("rb") as source:
            st.download_button("ดาวน์โหลด CSV ที่ส่งออกล่าสุด", source, file_name=DEFAULT_OUTPUT.name, mime="text/csv")
    st.code(r".\.venv\Scripts\python.exe 2_train_model.py --data dataset/video_pose_data.csv --model rf", language="powershell")
    st.caption("ป้ายกำกับเป็นสิ่งที่ผู้ใช้กำหนดเอง ระบบไม่ใช้กฎท่าทางมาตัดสินแทนผู้ติดป้าย")
