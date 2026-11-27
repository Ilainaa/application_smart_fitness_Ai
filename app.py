"""Smart Fitness AI app shell. Run: streamlit run app.py (Streamlit 1.45.1)."""
from html import escape
from pathlib import Path
import os
import uuid

import av
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as component_html
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from camera_pipeline import FitnessEngine, CaptureWorker, camera_source, load_model, ROOT
from fitness_core import EXERCISES, CLASSES

# Shared with the Labeler; importing this module does not start a session.
CSS = """<style>
:root {--bg:#071E22;--card:#123238;--mint:#6EE7B7;--text:#E5F5EF;
 --muted:#ACCCC3;--line:#426660;--red:#F87171;--amber:#FBBF24;--radius:24px;}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="stSidebarCollapsedControl"],
.stAppDeployButton, #MainMenu, footer {display:none !important;}
.stApp {background:var(--bg);color:var(--text);}
.stApp, button, input {font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Tahoma,sans-serif;}
.block-container {max-width:1320px;padding:24px 32px 48px;}
[data-testid="stMainBlockContainer"] {min-width:0;}
h1,h2,h3,p {overflow-wrap:anywhere;}
h1 {font-size:clamp(2rem,4vw,3rem) !important;letter-spacing:-.04em;line-height:1.2 !important;}
h2 {font-size:1.4rem !important;} h3 {font-size:1.1rem !important;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, .muted {color:var(--muted);}
button, a, input, summary {touch-action:manipulation;scroll-margin-block:90px;}
button, [data-testid="stPageLink"] a, [data-baseweb="tab"], summary {min-height:48px !important;}
button {border-radius:16px !important;transition:background .16s,box-shadow .16s,opacity .1s !important;}
button:hover:not(:disabled) {box-shadow:0 0 0 1px var(--mint);}
button:active:not(:disabled) {opacity:.75;}
button:focus-visible, a:focus-visible, summary:focus-visible, input:focus-visible,
[role="slider"]:focus-visible {outline:3px solid var(--mint) !important;outline-offset:3px;}
button[kind="primary"] {background:var(--mint);color:var(--bg);border-color:var(--mint);font-weight:700;}
button:disabled {opacity:.5;}
[data-testid="stExpander"] {background:var(--card);border-radius:16px;border-color:var(--line);}
[data-testid="stCheckbox"] label, [role="radiogroup"] label {min-height:48px;align-items:center;}
[data-baseweb="input"], [data-baseweb="select"] {min-height:48px;}
.app-header {display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:8px;}
.brand {display:flex;align-items:center;gap:12px;font-weight:750;font-size:1rem;letter-spacing:-.02em;}
.brand svg {width:40px;height:40px;background:var(--mint);color:var(--bg);border-radius:12px;padding:8px;}
.screen-name {font-size:.85rem;color:var(--muted);text-align:right;}
.st-key-app_nav {max-width:480px;margin:0 auto 16px;background:var(--card);border:1px solid var(--line);
 border-radius:20px;padding:6px;}
.st-key-app_nav [data-testid="stHorizontalBlock"] {flex-wrap:nowrap !important;gap:8px;}
.st-key-app_nav [data-testid="stColumn"] {min-width:0 !important;flex:1 1 0 !important;width:auto !important;}
.st-key-app_nav button {border:0;min-height:48px;}
.st-key-app_nav button p {font-size:.9rem;}
.kicker {color:var(--mint);font-size:.75rem;font-weight:750;letter-spacing:.15em;text-transform:uppercase;}
.intro {margin:0 0 8px;} .intro h1 {margin:8px 0 12px;padding:0;}
.intro p {color:var(--muted);margin:0;max-width:60ch;line-height:1.7;}
.steps {display:flex;gap:12px;margin-bottom:8px;}
.step {flex:1;border-top:2px solid var(--line);padding-top:12px;color:var(--muted);font-size:.85rem;}
.step.active {border-color:var(--mint);color:var(--text);}
.step b {display:inline-flex;background:var(--card);border-radius:50%;width:28px;height:28px;
 align-items:center;justify-content:center;margin-right:8px;color:var(--mint);}
.st-key-camera_toggle [role="radiogroup"] {gap:8px;}
.st-key-camera_toggle [role="radiogroup"] label {background:var(--card);border:1px solid var(--line);
 border-radius:16px;padding:8px 16px;flex:1;}
.st-key-camera_toggle label:has(input:checked) {border-color:var(--mint);}
.st-key-exercise_grid [class*="st-key-exercise_"] {background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
 padding:16px;box-shadow:0 8px 24px #00000018;}
.st-key-exercise_grid [class*="st-key-exercise_"]:has(button[kind="primary"]) {border-color:var(--mint);box-shadow:0 0 24px #6ee7b70d;}
.exercise-art {height:104px;display:flex;align-items:center;justify-content:center;
 background:radial-gradient(ellipse,#6ee7b712,transparent 70%);}
.exercise-art svg {width:112px;height:112px;color:var(--mint);}
.exercise-tip {min-height:48px;color:var(--muted);font-size:.85rem;margin:12px 0 0;line-height:1.5;}
.st-key-start_session button {min-height:64px !important;font-size:1.1rem;}
.st-key-start_session {margin-top:8px;}
.st-key-end_session button {min-height:56px !important;}
.note {border:1px solid var(--line);border-left:3px solid var(--mint);border-radius:16px;
 padding:16px 20px;background:var(--card);line-height:1.6;overflow-wrap:anywhere;}
.note.warn {border-left-color:var(--amber);} .note.error {border-left-color:var(--red);}
.note small {color:var(--muted);display:block;margin-top:4px;}
.live-title {display:flex;justify-content:space-between;align-items:center;gap:16px;margin:16px 0;}
.live-title h1 {margin:4px 0;padding:0;}
.live-tag {color:var(--mint);border:1px solid var(--line);border-radius:99px;padding:8px 12px;font-size:.75rem;}
.st-key-camera_view {background:#041316;border:1px solid var(--line);border-radius:var(--radius);
 padding:12px;overflow:hidden;}
.st-key-camera_view [data-testid="stImage"] img {max-height:68vh;object-fit:contain;border-radius:16px;}
.st-key-camera_view iframe {border-radius:16px;max-width:100%;}
.camera-empty {min-height:360px;display:flex;flex-direction:column;align-items:center;justify-content:center;
 text-align:center;gap:16px;padding:24px;color:var(--muted);}
.camera-empty svg {width:64px;height:64px;color:var(--mint);}
.hud {display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:center;}
.counter {text-align:center;} .counter strong {display:block;font-size:clamp(5rem,8vw,8rem);
 font-weight:800;line-height:1.05;letter-spacing:-.06em;font-variant-numeric:tabular-nums;}
.counter small {color:var(--muted);font-size:.85rem;}
.score-ring {position:relative;max-width:220px;width:100%;margin:auto;aspect-ratio:1;}
.score-ring svg {width:100%;height:100%;transform:rotate(-90deg);}
.score-ring .ring-label {position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;}
.ring-label strong {font-size:clamp(2.5rem,4.5vw,4.5rem);line-height:1.1;font-variant-numeric:tabular-nums;}
.ring-label small {color:var(--muted);font-size:.8rem;}
.feedback {border-radius:20px;border:1px solid currentColor;padding:20px;margin:16px 0;
 color:var(--mint);background:#6ee7b70b;font-size:clamp(1.15rem,2vw,1.55rem);font-weight:650;line-height:1.5;}
.feedback.warn {color:var(--amber);background:#fbbf240a;} .feedback.error {color:var(--red);background:#f871710a;}
.feedback small {display:block;font-size:.7rem;letter-spacing:.1em;margin-bottom:8px;}
.gauges {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;}
.gauge {background:var(--card);border-radius:20px;padding:12px;text-align:center;}
.gauge svg {display:block;width:100%;max-height:90px;} .gauge span {font-size:.85rem;color:var(--muted);}
.hud-detail {color:var(--muted);font-size:.8rem;line-height:1.7;margin-top:16px;overflow-wrap:anywhere;}
.summary-hero {display:flex;align-items:center;gap:24px;padding:32px;background:linear-gradient(120deg,var(--card),#0c282c);
 border:1px solid var(--line);border-radius:var(--radius);margin:16px 0 24px;}
.grade {width:112px;height:112px;flex-shrink:0;border:2px solid var(--mint);border-radius:50%;display:flex;
 align-items:center;justify-content:center;color:var(--mint);font-size:4.5rem;font-weight:800;box-shadow:0 0 32px #6ee7b711;}
.summary-hero h1 {margin:8px 0;padding:0;font-size:clamp(1.6rem,3vw,2.5rem) !important;}
.summary-hero p {color:var(--muted);margin:0;}
.stats {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-bottom:24px;}
.stat {background:var(--card);border-radius:20px;padding:24px;min-width:0;}
.stat strong {display:block;font-size:clamp(1.4rem,3vw,2.4rem);font-variant-numeric:tabular-nums;margin:8px 0;overflow-wrap:anywhere;}
.stat span {color:var(--muted);font-size:.85rem;}
.empty {text-align:center;padding:64px 24px;background:var(--card);border-radius:var(--radius);}
.empty svg {width:64px;height:64px;color:var(--mint);margin-bottom:16px;}
.improve {padding:24px;background:var(--card);border-radius:var(--radius);margin:16px 0;}
.improve p {margin:8px 0;color:var(--muted);line-height:1.7;}
.st-key-screen_choose, .st-key-screen_train, .st-key-screen_review {animation:screen-in .24s ease-out;}
@keyframes screen-in {from {opacity:.2;transform:translateY(8px);} to {opacity:1;transform:none;}}
@media (max-width:767px) {
 .block-container {padding:16px 16px calc(112px + env(safe-area-inset-bottom));}
 .app-header {padding-top:env(safe-area-inset-top);}.brand {font-size:.9rem;}.screen-name {font-size:.75rem;}
 .st-key-app_nav {position:fixed;bottom:0;left:0;right:0;max-width:none;margin:0;z-index:100;
 border-radius:20px 20px 0 0;border-bottom:0;padding:8px 16px calc(8px + env(safe-area-inset-bottom));
 box-shadow:0 -8px 32px #0005;background:var(--bg);}
 .st-key-app_nav button {flex-direction:column;gap:2px;padding:6px 2px;min-height:64px !important;}
 .st-key-app_nav button p {font-size:.8rem;white-space:nowrap;overflow-wrap:normal;line-height:1.25;}
 .intro {margin:16px 0;}.steps {gap:8px;}.step {font-size:.75rem;}.step b {display:flex;margin-bottom:8px;}
 .st-key-exercise_grid [data-testid="stHorizontalBlock"] {flex-wrap:wrap !important;gap:12px;}
 .st-key-exercise_grid [data-testid="stColumn"] {width:calc(50% - 6px) !important;flex:1 1 calc(50% - 6px) !important;min-width:0 !important;}
 [class*="st-key-exercise_"]:not(.st-key-exercise_grid) {padding:12px;}
 .exercise-art {height:96px;}.exercise-art svg {height:88px;}.exercise-tip {font-size:.8rem;min-height:60px;}
 .st-key-live_layout [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
 .st-key-live_layout [data-testid="stColumn"] {min-width:100% !important;}
 .st-key-camera_view [data-testid="stImage"] img {max-height:52vh;}.camera-empty {min-height:240px;}
 .counter strong {font-size:5.5rem;}.ring-label strong {font-size:3rem;}.score-ring {max-width:180px;}
 .stats {grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;}.stat {padding:20px;}
 .summary-hero {padding:24px 16px;gap:16px;}.grade {width:80px;height:80px;font-size:3.5rem;}
}
@media (prefers-reduced-motion:reduce) {*,*::before,*::after {animation:none !important;transition:none !important;}}
</style>"""

CAMERAS = ["มือถือ / Browser (WebRTC)", "DroidCam / IP Camera / Webcam"]
TIPS = {
    "squat": "เป้าหมาย: มุมเข่าที่จุดต่ำสุด 70–100° และมุมแนวหลัง Ear–Shoulder–Hip >150°",
    "pushup": "เป้าหมาย: มุมศอกจุดต่ำสุด 65–100° ลำตัวตรง ≥160° และอยู่ในแนวขนานพื้น",
    "plank": "นับเวลาเฉพาะช่วง Shoulder–Hip–Ankle 160–180° และลำตัวอยู่ในแนวขนานพื้น",
    "jj": "เป้าหมาย: ไหล่ทั้งสอง >140° ข้อมือเหนือศีรษะ และข้อเท้ากว้างกว่าช่วงไหล่",
}
SHORT_TIPS = {"squat": "กล้องด้านข้าง · ย่อแล้วกลับมายืน", "pushup": "กล้องด้านข้าง · รักษาลำตัวตรง",
              "plank": "กล้องด้านข้าง · จับเวลาท่าที่ถูกต้อง", "jj": "กล้องด้านหน้า · เห็นมือและเท้าครบ"}
POSES = {
    "squat": '<circle cx="62" cy="18" r="7"/><path d="M59 30 55 58 34 72 59 91M55 58 75 72 69 91M59 34 31 42 12 39M59 34 78 44 96 42M50 93h17M65 93h15"/>',
    "pushup": '<circle cx="86" cy="38" r="7"/><path d="m74 43-29 8-29 22h-9M71 45 79 61 69 77h17M44 51 22 66 14 77M9 84h85"/>',
    "plank": '<circle cx="85" cy="37" r="7"/><path d="m74 44-30 9-29 24h-8M72 45 70 75h22M44 53 30 67 19 77M7 84h88"/>',
    "jj": '<circle cx="55" cy="20" r="7"/><path d="M55 32v31M55 36 34 24 23 8M55 36 76 24 87 8M55 63 35 94H24M55 63 75 94h11"/>',
    "logo": '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    "camera": '<rect x="3" y="6" width="18" height="14" rx="3"/><circle cx="12" cy="13" r="4"/><path d="m8 6 2-3h4l2 3"/>',
}


def icon(name):
    view = "0 0 110 110" if name in EXERCISES else "0 0 24 24"
    return (f'<svg viewBox="{view}" fill="none" stroke="currentColor" stroke-width="2.5" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{POSES[name]}</svg>')


def html(content):
    st.markdown(content, unsafe_allow_html=True)


def notice(message, tone="", detail=""):
    html(f'<div class="note {tone}">{escape(message)}'
         + (f'<small>{escape(detail)}</small>' if detail else "") + '</div>')


def navigate(screen):
    st.session_state.screen = screen


def render_shell(screen):
    html(CSS)
    titles = {"choose": "เลือกการฝึก", "train": "ฝึกซ้อมสด", "review": "สรุปผล", "labeler": "นำเข้าวิดีโอ / ติดป้ายกำกับ"}
    html(f'<header class="app-header"><div class="brand">{icon("logo")}Smart Fitness AI</div>'
         f'<div class="screen-name">{titles[screen]}</div></header>')
    active = st.session_state.get("active", False)
    with st.container(key="app_nav"):
        train, review, labeler = st.columns(3)
        train_clicked = train.button("Train", icon=":material/fitness_center:", use_container_width=True,
                                     type="primary" if screen in ("choose", "train") else "secondary", disabled=active,
                                     on_click=navigate, args=("choose",), key="nav_train")
        review_clicked = review.button("Review", icon=":material/insights:", use_container_width=True,
                                       type="primary" if screen == "review" else "secondary", disabled=active,
                                       on_click=navigate, args=("review",), key="nav_review")
        if labeler.button("Labeler", icon=":material/video_library:", use_container_width=True,
                          type="primary" if screen == "labeler" else "secondary", disabled=active or screen == "labeler",
                          help="นำเข้าวิดีโอ / ติดป้ายกำกับ", key="nav_labeler"):
            st.switch_page("pages/1_Video_Labeler.py")
        if screen == "labeler" and (train_clicked or review_clicked):
            st.switch_page("app.py")


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
    st.session_state.screen = "review"


def choose_exercise(exercise):
    st.session_state.preferences["exercise"] = exercise


def start_session(artifact):
    prefs = st.session_state.preferences
    st.session_state.pop("setup_error", None)
    try:
        source = None if prefs["camera"] == CAMERAS[0] else camera_source(prefs["source"])
        engine = FitnessEngine(prefs["exercise"], None if prefs["rule_only"] else artifact, prefs["confidence"])
        st.session_state.engine = engine
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.session_mode = prefs["camera"]
        st.session_state.session_rule_only = prefs["rule_only"]
        st.session_state.worker = None if source is None else CaptureWorker(source, engine)
        st.session_state.active = True
        st.session_state.review = None
        st.session_state.screen = "train"
    except (ValueError, RuntimeError) as exc:
        st.session_state.setup_error = str(exc)


def choose_screen(artifact, model_error):
    prefs = st.session_state.preferences
    html('<div class="intro"><div class="kicker">YOUR DAILY MOVEMENT</div><h1>เลือกการฝึก</h1>'
         '<p>ฝึกอย่างเข้าใจท่าทาง • AI + Joint Geometry • Live Coaching</p></div>'
         '<div class="steps"><div class="step active"><b>1</b>ตั้งกล้อง</div>'
         '<div class="step active"><b>2</b>เลือกท่า</div><div class="step"><b>3</b>เริ่มฝึก</div></div>')
    with st.container(key="camera_toggle"):
        prefs["camera"] = st.radio("โหมดกล้อง", CAMERAS, index=CAMERAS.index(prefs["camera"]), horizontal=True, key="camera_choice")
    st.subheader("ท่าออกกำลังกาย")
    with st.container(key="exercise_grid"):
        for col, exercise in zip(st.columns(4), EXERCISES):
            with col, st.container(key=f"exercise_{exercise}"):
                html(f'<div class="exercise-art">{icon(exercise)}</div>')
                selected = prefs["exercise"] == exercise
                st.button(EXERCISES[exercise], icon=":material/check_circle:" if selected else None,
                          type="primary" if selected else "secondary", use_container_width=True,
                          on_click=choose_exercise, args=(exercise,), key=f"pick_{exercise}")
                html(f'<p class="exercise-tip">{SHORT_TIPS[exercise]}</p>')
    exercise = prefs["exercise"]
    st.caption(TIPS[exercise])
    with st.expander("ตั้งค่าการฝึก · Settings", icon=":material/tune:"):
        prefs["source"] = st.text_input("Camera index หรือ URL", prefs["source"],
                                       disabled=prefs["camera"] == CAMERAS[0], key="camera_source")
        prefs["confidence"] = st.slider("เกณฑ์ความมั่นใจ AI", .4, .95, prefs["confidence"], .05, key="ai_confidence")
        prefs["rule_only"] = st.checkbox("ทดสอบด้วยกฎอย่างเดียว (ไม่ใช้ Custom AI)", value=prefs["rule_only"], key="rule_only")
        if artifact:
            st.caption(f"Held-out accuracy: {artifact['accuracy']:.1%} · {artifact['model_type'].upper()}")
        st.caption("Squat / Push-up / Plank: กล้องด้านข้าง\n\nJumping Jacks: กล้องด้านหน้า เห็นศีรษะ มือ และเท้าครบ")
    missing = [] if artifact is None else sorted({c for c in CLASSES if c.startswith(exercise + "_")} - set(artifact["classes"]))
    if prefs["rule_only"]:
        notice("โหมด Rule-only: ผลลัพธ์ไม่ได้ผ่านการตรวจสอบจากโมเดลที่เทรนเอง", "warn")
    elif model_error:
        notice(f"โหลดโมเดลไม่ได้: {model_error}", "error")
    elif artifact is None:
        notice("ยังไม่มีโมเดล: เก็บข้อมูล แล้วรัน 2_train_model.py", "warn")
    elif missing:
        notice("โมเดลยังขาดคลาสของท่านี้: " + ", ".join(missing), "warn")
    else:
        notice(f"Custom model พร้อมใช้งาน · {len(artifact['classes'])} classes")
    can_start = prefs["rule_only"] or (artifact is not None and not missing)
    if st.session_state.get("setup_error"):
        notice(st.session_state.setup_error, "error")
    with st.container(key="start_session"):
        st.button("เริ่มเซสชันใหม่", icon=":material/play_arrow:", type="primary", use_container_width=True,
                  disabled=not can_start, key="start", on_click=start_session, args=(artifact,))


def gauge(label, value):
    progress = max(0, min(100, value / 180 * 100)) if value is not None else 0
    number = f"{value:.0f}°" if value is not None else "—"
    return (f'<div class="gauge"><svg viewBox="0 0 160 100" role="img" aria-label="{label} {number}">'
            '<path d="M20 75 A60 60 0 0 1 140 75" fill="none" stroke="var(--line)" stroke-width="8"/>'
            f'<path d="M20 75 A60 60 0 0 1 140 75" fill="none" stroke="var(--mint)" stroke-width="8" '
            f'pathLength="100" stroke-dasharray="{progress} 100" stroke-linecap="round"/>'
            f'<text x="80" y="76" text-anchor="middle" fill="var(--text)" font-size="27" font-weight="700">{number}</text>'
            f'</svg><span>{label}</span></div>')


def render_hud(snap):
    metrics = snap["metrics"]
    has_pose = bool(metrics)
    score = max(0, min(100, snap["score"]))
    count = f"{snap['seconds']:.1f}" if snap["exercise"] == "plank" else str(snap["reps"])
    count_label = "เวลาที่ท่าถูกต้อง · วินาที" if snap["exercise"] == "plank" else "จำนวนครั้งทั้งหมด"
    score_text = f"{score:.0f}" if has_pose else "—"
    html(f'<div class="hud"><div class="counter"><strong>{count}</strong><small>{count_label}</small></div>'
         '<div class="score-ring"><svg viewBox="0 0 120 120" aria-hidden="true">'
         '<circle cx="60" cy="60" r="52" fill="none" stroke="var(--line)" stroke-width="7"/>'
         f'<circle cx="60" cy="60" r="52" fill="none" stroke="var(--mint)" stroke-width="7" '
         f'pathLength="100" stroke-dasharray="{score if has_pose else 0} 100" stroke-linecap="round"/></svg>'
         f'<div class="ring-label"><strong>{score_text}<span style="font-size:.4em">%</span></strong>'
         '<small>Form Score ขณะนี้</small></div></div></div>')
    tone = "" if snap["valid"] else ("error" if has_pose else "warn")
    status = "ท่าทางถูกต้อง" if snap["valid"] else ("ปรับท่าทาง" if has_pose else "รอภาพที่เห็นท่าชัดเจน")
    html(f'<div class="feedback {tone}"><small>{status}</small>{escape(snap["feedback"])}</div>')
    keys = {"squat": [("เข่า", "knee"), ("หลัง", "back")],
            "pushup": [("ข้อศอก", "elbow"), ("แนวลำตัว", "body")],
            "plank": [("แนวลำตัว", "body"), ("ข้อศอก", "elbow")],
            "jj": [("ไหล่ซ้าย", "shoulder_left"), ("ไหล่ขวา", "shoulder_right")]}
    html('<div class="gauges">' + ''.join(gauge(label, metrics.get(key)) for label, key in keys[snap["exercise"]]) + '</div>')
    average = f"{snap['average_score']:.1f}%" if snap["frames"] else "—"
    detail = f"Form Score เฉลี่ย {average} · สถานะ {snap['state']}"
    if snap["exercise"] != "plank":
        detail += f" · ครั้งที่ท่าถูกต้อง {snap['correct_reps']}"
    if metrics:
        detail += (f" · ไหล่ซ้าย/ขวา {metrics['shoulder_left']:.0f}° / {metrics['shoulder_right']:.0f}°"
                   f" · ระยะข้อเท้า/ไหล่ {metrics['ankle_ratio']:.2f} · AI: {snap['ai_label']}")
        if snap["ai_label"] != "Rule-only":
            detail += f" ({snap['confidence']:.0%})"
    html(f'<div class="hud-detail">{escape(detail)}</div>')
    if snap["error"]:
        notice(snap["error"], "error")
    if st.session_state.engine.closed:
        notice("กล้องหยุดแล้ว กดจบเซสชันเพื่อบันทึกสรุป ก่อนเริ่มเซสชันใหม่", "warn")


@st.fragment(run_every=0.2)
def live_dashboard(browser_mode):
    # Camera decoding/inference stays in the existing processor/worker, outside the UI fragment.
    worker = st.session_state.get("worker")
    if browser_mode:
        render_hud(st.session_state.engine.snapshot())
    else:
        with st.container(key="live_layout"):
            camera, hud = st.columns([1.6, 1], gap="large")
            with camera, st.container(key="camera_view"):
                frame = worker.latest()
                if frame is not None:
                    st.image(frame, channels="BGR", use_container_width=True)
                else:
                    html(f'<div class="camera-empty">{icon("camera")}<span>กำลังเชื่อมต่อกล้อง…</span></div>')
                if worker.error:
                    notice(worker.error, "error")
            with hud:
                render_hud(st.session_state.engine.snapshot())


def train_screen():
    engine = st.session_state.engine
    exercise = st.session_state.preferences["exercise"]
    html(f'<div class="live-title"><div><div class="kicker">TRAIN / ฝึกซ้อมสด</div>'
         f'<h1>{escape(EXERCISES[exercise])}</h1></div><span class="live-tag">LIVE SESSION</span></div>')
    with st.container(key="end_session"):
        st.button("จบเซสชันและดูสรุป", icon=":material/stop:", type="primary", use_container_width=True,
                  key="end", on_click=finish_session)
    browser_mode = st.session_state.session_mode == CAMERAS[0]
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
        with st.container(key="live_layout"):
            camera, hud = st.columns([1.6, 1], gap="large")
            with camera, st.container(key="camera_view"):
                webrtc_streamer(key=st.session_state.session_id, mode=WebRtcMode.SENDRECV,
                                video_processor_factory=PoseProcessor, async_processing=False,
                                video_html_attrs={"autoPlay": True, "controls": False, "muted": True, "playsInline": True,
                                                  "style": {"width": "100%", "maxHeight": "640px", "aspectRatio": "4 / 3",
                                                            "objectFit": "contain", "borderRadius": "16px"}},
                                media_stream_constraints={"video": {"facingMode": {"ideal": "environment"},
                                                                     "width": {"ideal": 640}, "height": {"ideal": 480},
                                                                     "frameRate": {"ideal": 15, "max": 24}}, "audio": False},
                                rtc_configuration={"iceServers": ice_servers})
            with hud:
                live_dashboard(True)
    else:
        live_dashboard(False)


def review_screen():
    review = st.session_state.review
    if not review:
        html(f'<div class="empty">{icon("logo")}<h1>สรุปผล</h1>'
             '<p class="muted">ผลลัพธ์และกราฟจะปรากฏหลังจบเซสชัน</p></div>')
        st.button("เริ่มเซสชันใหม่", type="primary", use_container_width=True, on_click=navigate, args=("choose",))
        return
    grade = ("A" if review["average_score"] >= 85 else "B" if review["average_score"] >= 70 else "C") if review["frames"] else "—"
    mode = "Rule-only" if st.session_state.session_rule_only else "Custom AI + Rules"
    html(f'<div class="summary-hero"><div class="grade" aria-label="Performance Grade {grade}">{grade}</div>'
         f'<div><div class="kicker">WORKOUT COMPLETE</div><h1>สรุปผล · {EXERCISES[review["exercise"]]}</h1>'
         f'<p>{mode} · Performance Grade</p></div></div>')
    stats = [("เวลาที่ท่าถูกต้อง", f"{review['seconds']:.1f} วินาที") if review["exercise"] == "plank" else ("จำนวนครั้งทั้งหมด", str(review["reps"])),
             ("ครั้งที่ท่าถูกต้อง", str(review["correct_reps"]) if review["exercise"] != "plank" else "—"),
             ("Form Score เฉลี่ย", f"{review['average_score']:.1f}%" if review["frames"] else "—"),
             ("สถานะ", review["state"])]
    html('<div class="stats">' + ''.join(f'<div class="stat"><span>{label}</span><strong>{escape(value)}</strong></div>' for label, value in stats) + '</div>')
    st.caption("เกณฑ์แสดงผล: A ≥85% · B ≥70% · C <70% จาก Form Score เฉลี่ย")
    if not review["frames"]:
        notice("ไม่มีเฟรมที่ประเมินได้ในเซสชันนี้", "warn")
    history = pd.DataFrame(review["history"])
    if not history.empty:
        left, right = st.columns(2)
        with left:
            st.caption("Form Score ตลอดเซสชัน")
            st.line_chart(history.set_index("elapsed_s")[["form_score"]], color="#6EE7B7", height=220, x_label="วินาที", y_label="Form Score (%)")
        with right:
            st.caption("เวลาสะสม / จำนวนครั้งสะสม")
            columns = ["plank_s"] if review["exercise"] == "plank" else ["reps", "correct_reps"]
            st.line_chart(history.set_index("elapsed_s")[columns], color=["#6EE7B7"] if len(columns) == 1 else ["#6EE7B7", "#FBBF24"],
                          height=220, x_label="วินาที", y_label="วินาทีสะสม" if review["exercise"] == "plank" else "ครั้ง")
    notes = []
    if not review["frames"]:
        notes.append("จัดกล้องให้เห็นร่างกายครบ แล้วลองเริ่มเซสชันใหม่")
    else:
        if review["exercise"] != "plank" and review["reps"] > review["correct_reps"]:
            notes.append(f"มี {review['reps'] - review['correct_reps']} ครั้งที่ยังไม่ผ่านเกณฑ์ท่าถูกต้องครบทุกข้อ")
        if review["exercise"] != "plank" and not review["reps"]:
            notes.append("ยังไม่มีรอบที่นับครบ ลองทำท่าจากจุดเริ่มต้นและกลับมาจุดเริ่มต้นให้ครบหนึ่งรอบ")
        if review["average_score"] < 85:
            notes.append("ฝึกช้า ๆ และใช้คำแนะนำบนหน้าจอเพื่อรักษาท่าทางให้สม่ำเสมอ")
        else:
            notes.append("รักษาความสม่ำเสมอของท่าทางในเซสชันถัดไป")
        notes.append(TIPS[review["exercise"]])
    html('<div class="improve"><h2>จุดที่ควรปรับปรุง</h2>' + ''.join(f'<p>{escape(note)}</p>' for note in notes)
         + '<p style="font-size:.8rem">คำแนะนำจากคะแนนและจำนวนครั้งรวม ระบบไม่ได้บันทึกประเภทข้อผิดพลาดรายครั้ง</p></div>')
    if not history.empty:
        st.download_button("ดาวน์โหลดผล CSV", history.to_csv(index=False).encode("utf-8-sig"), file_name="fitness_session.csv",
                           mime="text/csv", icon=":material/download:", use_container_width=True)
    st.button("เริ่มเซสชันใหม่", icon=":material/replay:", type="primary", use_container_width=True, on_click=navigate, args=("choose",))
    st.caption("Form Score = ค่าเฉลี่ยคะแนนเงื่อนไขต่อเฟรมที่เห็นท่าชัดเจน ไม่ใช่ Accuracy ของโมเดล "
               "· กราฟเก็บสูงสุด 18,000 จุดล่าสุด แต่ตัวนับและคะแนนเฉลี่ยครอบคลุมทั้งเซสชัน")


def main():
    st.set_page_config(page_title="Smart Fitness AI", page_icon=":material/fitness_center:", layout="wide", initial_sidebar_state="collapsed")
    st.session_state.setdefault("active", False)
    st.session_state.setdefault("review", None)
    st.session_state.setdefault("screen", "choose")
    st.session_state.setdefault("preferences", {"exercise": "squat", "camera": CAMERAS[0], "source": "0", "confidence": .6, "rule_only": False})
    if st.session_state.active:
        st.session_state.screen = "train"
    elif st.session_state.screen == "train":
        st.session_state.screen = "choose"
    screen = st.session_state.screen
    render_shell(screen)
    with st.container(key=f"screen_{screen}"):
        if screen == "choose":
            artifact, model_error = None, None
            model_path = ROOT / "models/fitness_model.pkl"
            if model_path.exists():
                try:
                    artifact = cached_model(str(model_path), model_path.stat().st_mtime_ns)
                except Exception as exc:
                    model_error = str(exc)
            choose_screen(artifact, model_error)
        elif screen == "train":
            train_screen()
        else:
            review_screen()
    if st.session_state.get("previous_screen") != screen:
        # shortcut: Streamlit has no scroll-to-top API; revisit this DOM hook on framework upgrades.
        component_html(f"<!-- {screen} -->" + """<script>
        const main = window.parent.document.querySelector('[data-testid="stMain"]');
        requestAnimationFrame(() => { if (main) main.scrollTo({top: 0, behavior: 'instant'}); });
        </script>""", height=0)
        st.session_state.previous_screen = screen


if __name__ == "__main__":
    main()
