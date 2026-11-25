"""Compact frame review, mounted inside the existing video annotation page."""
import math
import streamlit as st
from fitness_core import CLASSES, EXERCISES
from frame_labeling import generate_frames, label_frames, review_image

LABEL_NAMES = {
    "squat_up": "ยืน", "squat_down_correct": "ย่อถูกต้อง", "squat_down_wrong": "ย่อผิด",
    "pushup_up": "เหยียดแขน", "pushup_down_correct": "ลดตัวถูกต้อง", "pushup_down_wrong": "ลดตัวผิด",
    "plank_correct": "แพลงก์ถูกต้อง", "plank_wrong": "แพลงก์ผิด",
    "jj_closed": "แขนลง เท้าชิด", "jj_open_correct": "กางถูกต้อง", "jj_open_wrong": "กางผิด",
    "skip": "ข้ามภาพ", None: "ยังไม่ติดป้าย",
}


def label_name(value):
    return LABEL_NAMES.get(value, value)


def _move_page(key, step):
    st.session_state[key] += step


def show_frame_labeler(project_path, project, clip):
    frame_set = clip.get("frame_set")
    with st.expander("ตั้งค่าการแตกภาพ", expanded=not bool(frame_set)):
        with st.form(f"generate_{clip['id']}"):
            left, right = st.columns(2)
            exercise = left.selectbox("ท่าของชุดภาพ", list(EXERCISES), format_func=EXERCISES.get,
                                     index=list(EXERCISES).index(frame_set["exercise"]) if frame_set else 0)
            interval = right.number_input("ดึงภาพทุกกี่วินาที", min_value=.1, max_value=10.0,
                                          value=frame_set["interval"] if frame_set else 1.0, step=.1)
            st.caption("เริ่มที่ 1 วินาทีต่อภาพ • ท่าที่เคลื่อนไหวเร็ว ใช้ 0.2–0.5 วินาที")
            replace = st.checkbox("ยืนยันสร้างชุดภาพใหม่แทนชุดและป้ายรายภาพเดิม") if frame_set else True
            generate = st.form_submit_button("แตกภาพและตรวจจับข้อต่อ", type="primary")
        if generate:
            if not replace:
                st.warning("กรุณายืนยันก่อนแทนชุดภาพเดิม")
            else:
                bar = st.progress(0.0)
                try:
                    count = generate_frames(project_path, clip["id"], project["revision"], exercise, interval,
                                            progress=lambda p: bar.progress(p))
                    st.session_state.video_notice = f"สร้าง {count} ภาพแล้ว พร้อมติดป้าย"
                    st.rerun()
                except Exception as exc:
                    st.error(f"สร้างภาพไม่สำเร็จ: {exc}")
    if not frame_set:
        st.info("เลือกท่าแล้วกดแตกภาพ เพื่อเริ่มตรวจและติดป้าย")
        return
    if frame_set["rotation"] != clip.get("rotation", 0):
        st.warning("มีการเปลี่ยนแนวภาพ กรุณาสร้างชุดภาพใหม่ก่อนติดป้ายและส่งออก")
        return
    frames = frame_set["frames"]
    if not frames:
        st.info("ยังไม่มีภาพในชุดนี้ กรุณาสร้างชุดภาพใหม่")
        return
    labelled = sum(f["label"] in CLASSES for f in frames)
    skipped = sum(f["label"] == "skip" for f in frames)
    remaining = len(frames) - labelled - skipped
    for col, title, value in zip(st.columns(4), ["ภาพทั้งหมด", "ติดป้ายแล้ว", "ข้ามแล้ว", "รอตรวจ"],
                                 [len(frames), labelled, skipped, remaining]):
        col.metric(title, value)
    st.progress((labelled + skipped) / len(frames), text=f"ตรวจแล้ว {labelled + skipped} / {len(frames)} ภาพ")
    controls, settings = st.columns([2, 1])
    pages = math.ceil(len(frames) / 6)
    page_key = f"frame_page_{clip['id']}_{frames[0]['image']}"
    page = controls.number_input(f"หน้าภาพ (ทั้งหมด {pages} หน้า)", min_value=1, max_value=pages,
                                  value=1, step=1, key=page_key)
    overlay = settings.checkbox("แสดงจุดข้อต่อ", value=True)
    shown = frames[(page - 1) * 6:page * 6]
    st.markdown(f"**ภาพ {(page - 1) * 6 + 1}–{min(page * 6, len(frames))} จาก {len(frames)} ภาพ**"
                f" · เวลา {shown[0]['time']:.1f}–{shown[-1]['time']:.1f} วินาที")
    st.caption("1. เลือกภาพ → 2. เลือกป้าย → 3. บันทึกก่อนเปลี่ยนหน้า")
    with st.form(f"labels_{clip['id']}_{project['revision']}_{page}"):
        selected = []
        for start in range(0, len(shown), 3):
            for col, frame in zip(st.columns(3), shown[start:start + 3]):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{frame['time']:.1f} วินาที**")
                        try:
                            st.image(review_image(project_path, frame, overlay), channels="BGR", use_container_width=True)
                        except ValueError as exc:
                            st.error(str(exc))
                        st.caption(label_name(frame["label"]))
                        if frame["features"] is None:
                            st.caption("ข้อต่อไม่ครบ — เลือกข้ามภาพนี้")
                        if st.checkbox(f"เลือกภาพ {frame['index']}", key=f"pick_{clip['id']}_{project['revision']}_{frame['index']}"):
                            selected.append(frame["index"])
        st.divider()
        labels = [c for c in CLASSES if c.startswith(frame_set["exercise"] + "_")] + ["skip", None]
        label = st.selectbox("ป้ายสำหรับภาพที่เลือก", labels,
                             format_func=lambda v: "ล้างป้าย" if v is None else label_name(v))
        if st.form_submit_button("บันทึกป้ายภาพที่เลือก", type="primary", use_container_width=True):
            try:
                label_frames(project_path, clip["id"], project["revision"], selected, label)
                st.session_state.video_notice = f"บันทึก {len(selected)} ภาพแล้ว · {label_name(label) if label is not None else 'ล้างป้าย'}"
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    previous, counter, following = st.columns([1, 2, 1])
    previous.button("← หน้าก่อน", disabled=page == 1, on_click=_move_page, args=(page_key, -1),
                    use_container_width=True)
    counter.markdown(f"<div style='text-align:center;padding:10px'>หน้า {page} / {pages}</div>", unsafe_allow_html=True)
    following.button("หน้าถัดไป →", disabled=page == pages, on_click=_move_page, args=(page_key, 1),
                     use_container_width=True)
    st.caption("เส้น Skeleton แสดงจุดที่ตรวจพบ ไม่ใช่ผลตัดสินว่าท่าถูกต้อง • ตัวเลือกที่ยังไม่บันทึกจะไม่ถูกเก็บเมื่อเปลี่ยนหน้า")
