"""Smart Fitness AI: Choose -> Train -> Review. Run: streamlit run app.py"""
from pathlib import Path
import os
import uuid
import av
import pandas as pd
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from camera_pipeline import FitnessEngine, CaptureWorker, camera_source, load_model, ROOT
from fitness_core import EXERCISES, CLASSES

st.set_page_config(page_title="Smart Fitness AI", page_icon="💪", layout="wide")
st.markdown("""<style>
.block-container {max-width:1240px; padding-top:2rem;}
h1 {color:#baf8df; letter-spacing:-1px;}
[data-testid="stMetric"] {background:#123238; border:1px solid #28534f;
border-radius:16px; padding:18px;}
.hero {background:linear-gradient(115deg,#123c40,#0a252b);border:1px solid #2c5c55;
border-radius:24px;padding:28px;margin-bottom:22px;}
.eyebrow {color:#6ee7b7;font-size:13px;letter-spacing:3px;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def cached_model(path, modified_ns):
    return load_model(Path(path))


def finish_session():
    engine = st.session_state.engine
    worker = st.session_state.get("worker")
    if worker:
        worker.stop()
    engine.close()
    st.session_state.review = engine.snapshot()
    st.session_state.active = False
    st.session_state.worker = None


def show_metrics(snap):
    cols = st.columns(4)
    if snap["exercise"] == "plank":
        cols[0].metric("เวลาที่ท่าถูกต้อง", f"{snap['seconds']:.1f} วินาที")
    else:
        cols[0].metric("จำนวนครั้งทั้งหมด", snap["reps"])
    cols[1].metric("ครั้งที่ท่าถูกต้อง", snap["correct_reps"] if snap["exercise"] != "plank" else "—")
    cols[2].metric("Form Score เฉลี่ย", f"{snap['average_score']:.1f}%" if snap["frames"] else "—")
    cols[3].metric("สถานะ", snap["state"])


st.session_state.setdefault("active", False)
st.session_state.setdefault("review", None)
active = st.session_state.active
st.markdown('<div class="hero"><div class="eyebrow">MOVE WITH INTENTION</div>'
            '<h1>Smart Fitness AI</h1><p>ฝึกอย่างเข้าใจท่าทาง • AI + Joint Geometry • Live Coaching</p></div>', unsafe_allow_html=True)

artifact, model_error = None, None
model_path = ROOT / "models/fitness_model.pkl"
if model_path.exists():
    try:
        artifact = cached_model(str(model_path), model_path.stat().st_mtime_ns)
    except Exception as exc:
        model_error = str(exc)

with st.sidebar:
    st.title("ตั้งค่าการฝึก")
    st.page_link("pages/1_Video_Labeler.py", label="นำเข้าวิดีโอ / ติดป้ายกำกับ", icon="🎬", disabled=active)
    camera_mode = st.radio("โหมดกล้อง", ["มือถือ / Browser (WebRTC)", "DroidCam / IP Camera / Webcam"], disabled=active)
    exercise = st.selectbox("ท่าออกกำลังกาย", list(EXERCISES), format_func=EXERCISES.get, disabled=active)
    source_text = st.text_input("Camera index หรือ URL", "0", disabled=active or camera_mode.startswith("มือถือ"))
    confidence = st.slider("เกณฑ์ความมั่นใจ AI", 0.4, 0.95, 0.6, 0.05, disabled=active)
    rule_only = st.checkbox("ทดสอบด้วยกฎอย่างเดียว (ไม่ใช้ Custom AI)", value=False, disabled=active)
    st.divider()
    if artifact:
        st.success(f"Custom model พร้อมใช้งาน · {len(artifact['classes'])} classes")
        st.caption(f"Held-out accuracy: {artifact['accuracy']:.1%} · {artifact['model_type'].upper()}")
    elif model_error:
        st.error(f"โหลดโมเดลไม่ได้: {model_error}")
    else:
        st.warning("ยังไม่มีโมเดล: เก็บข้อมูล แล้วรัน 2_train_model.py")
    st.caption("Squat / Push-up / Plank: กล้องด้านข้าง\n\nJumping Jacks: กล้องด้านหน้า เห็นศีรษะ มือ และเท้าครบ")

st.subheader("01 / Choose — เลือกการฝึก")
st.write(f"**{EXERCISES[exercise]}** · {camera_mode}")
if exercise == "squat":
    st.caption("เป้าหมาย: มุมเข่าที่จุดต่ำสุด 70–100° และมุมแนวหลัง Ear–Shoulder–Hip >150°")
elif exercise == "pushup":
    st.caption("เป้าหมาย: มุมศอกจุดต่ำสุด 65–100° ลำตัวตรง ≥160° และอยู่ในแนวขนานพื้น")
elif exercise == "plank":
    st.caption("นับเวลาเฉพาะช่วง Shoulder–Hip–Ankle 160–180° และลำตัวอยู่ในแนวขนานพื้น")
else:
    st.caption("เป้าหมาย: ไหล่ทั้งสอง >140° ข้อมือเหนือศีรษะ และข้อเท้ากว้างกว่าช่วงไหล่")

missing_classes = [] if artifact is None else sorted(
    {c for c in CLASSES if c.startswith(exercise + "_")} - set(artifact["classes"]))
can_start = rule_only or (artifact is not None and not missing_classes)
if missing_classes and not rule_only:
    st.warning("โมเดลยังขาดคลาสของท่านี้: " + ", ".join(missing_classes))
if rule_only:
    st.info("โหมด Rule-only: ผลลัพธ์ไม่ได้ผ่านการตรวจสอบจากโมเดลที่เทรนเอง")
if st.button("เริ่มเซสชันใหม่", type="primary", disabled=active or not can_start):
    try:
        source = None if camera_mode.startswith("มือถือ") else camera_source(source_text)
        engine = FitnessEngine(exercise, None if rule_only else artifact, confidence)
        st.session_state.engine = engine
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.session_mode = camera_mode
        st.session_state.session_rule_only = rule_only
        st.session_state.worker = None if source is None else CaptureWorker(source, engine)
        st.session_state.active = True
        st.session_state.review = None
        st.rerun()
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))

st.subheader("02 / Train — ฝึกซ้อมสด")
if st.session_state.active:
    engine = st.session_state.engine
    if st.button("จบเซสชันและดูสรุป", type="primary"):
        finish_session()
        st.rerun()
    browser_mode = st.session_state.session_mode.startswith("มือถือ")
    if browser_mode:
        st.caption("กด START ด้านล่างและอนุญาตกล้อง · มือถือต้องเปิดผ่าน HTTPS; localhost ใช้ได้บนเครื่องเดียวกัน")

        class PoseProcessor(VideoProcessorBase):
            def __init__(self):
                self.engine = engine

            def recv(self, frame):
                output = self.engine.process(frame.to_ndarray(format="bgr24"))
                return av.VideoFrame.from_ndarray(output, format="bgr24")

            def on_ended(self):
                self.engine.close()

        ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
        if os.environ.get("TURN_URL"):
            ice_servers.append({"urls": [os.environ["TURN_URL"]], "username": os.environ.get("TURN_USERNAME", ""),
                                "credential": os.environ.get("TURN_PASSWORD", "")})
        webrtc_streamer(key=st.session_state.session_id, mode=WebRtcMode.SENDRECV,
                        video_processor_factory=PoseProcessor, async_processing=False,
                        media_stream_constraints={"video": {"facingMode": {"ideal": "environment"},
                                                             "width": {"ideal": 640}, "height": {"ideal": 480},
                                                             "frameRate": {"ideal": 15, "max": 24}}, "audio": False},
                        rtc_configuration={"iceServers": ice_servers})

    @st.fragment(run_every=0.2)
    def live_dashboard():
        worker = st.session_state.get("worker")
        if worker:
            frame = worker.latest()
            if frame is not None:
                st.image(frame, channels="BGR", use_container_width=True)
            else:
                st.info("กำลังเชื่อมต่อกล้อง…")
            if worker.error:
                st.error(worker.error)
        snap = st.session_state.engine.snapshot()
        show_metrics(snap)
        (st.success if snap["valid"] else st.warning)(snap["feedback"])
        st.progress(min(1.0, max(0.0, snap["score"] / 100)), text=f"Form Score ขณะนี้ {snap['score']:.0f}%")
        metrics = snap["metrics"]
        if metrics:
            cols = st.columns(4)
            for col, (label, key) in zip(cols, [("เข่า", "knee"), ("ข้อศอก", "elbow"), ("แนวลำตัว", "body"), ("หลัง", "back")]):
                col.metric(label, f"{metrics[key]:.0f}°")
            st.caption(f"ไหล่ซ้าย/ขวา {metrics['shoulder_left']:.0f}° / {metrics['shoulder_right']:.0f}° · "
                       f"ระยะข้อเท้า/ไหล่ {metrics['ankle_ratio']:.2f} · AI: {snap['ai_label']} "
                       + (f"({snap['confidence']:.0%})" if snap["ai_label"] != "Rule-only" else ""))
        if snap["error"]:
            st.error(snap["error"])
        if st.session_state.engine.closed:
            st.info("กล้องหยุดแล้ว กดจบเซสชันเพื่อบันทึกสรุป ก่อนเริ่มเซสชันใหม่")
    live_dashboard()
else:
    st.info("เลือกท่าและโหมดกล้อง แล้วกดเริ่มเซสชันใหม่")

st.subheader("03 / Review — สรุปผล")
review = st.session_state.review
if review:
    st.write(f"**{EXERCISES[review['exercise']]}** · " + ("Rule-only" if st.session_state.session_rule_only else "Custom AI + Rules"))
    show_metrics(review)
    if not review["frames"]:
        st.warning("ไม่มีเฟรมที่ประเมินได้ในเซสชันนี้")
    history = pd.DataFrame(review["history"])
    if not history.empty:
        left, right = st.columns(2)
        with left:
            st.caption("Form Score ตลอดเซสชัน")
            st.line_chart(history.set_index("elapsed_s")[["form_score"]], color="#6EE7B7")
        with right:
            st.caption("เวลาสะสม / จำนวนครั้งสะสม")
            columns = ["plank_s"] if review["exercise"] == "plank" else ["reps", "correct_reps"]
            st.line_chart(history.set_index("elapsed_s")[columns])
        st.download_button("ดาวน์โหลดผล CSV", history.to_csv(index=False).encode("utf-8-sig"),
                           file_name="fitness_session.csv", mime="text/csv")
    st.caption("Form Score = ค่าเฉลี่ยคะแนนเงื่อนไขต่อเฟรมที่เห็นท่าชัดเจน ไม่ใช่ Accuracy ของโมเดล "
               "· กราฟเก็บสูงสุด 18,000 จุดล่าสุด แต่ตัวนับและคะแนนเฉลี่ยครอบคลุมทั้งเซสชัน")
else:
    st.caption("ผลลัพธ์และกราฟจะปรากฏหลังจบเซสชัน")
