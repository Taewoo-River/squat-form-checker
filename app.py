"""Squat Form Checker -- Streamlit web application.

Run with:  streamlit run app.py

Analyse uploaded squat videos with MediaPipe pose estimation to count reps,
score form, and provide coaching feedback.
"""

import streamlit as st

# ─── page config (must be the first Streamlit call) ──────────────────
st.set_page_config(
    page_title="Squat Form Checker",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Squat Form Checker")
st.caption(
    "Real-time squat posture analysis and rep counting powered by MediaPipe pose estimation."
)

try:
    import os
    import tempfile

    import cv2
    import numpy as np

    _deps_ok = True
except Exception as e:
    _deps_ok = False
    st.error(
        "Could not load app dependencies (OpenCV, NumPy, or the squat_form_checker package). "
        f"Details: `{e!r}`"
    )
    st.info(
        "From the folder that contains `requirements.txt` run:\n\n"
        "`pip install -r requirements.txt`\n\n"
        "Then restart Streamlit."
    )

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── helpers ─────────────────────────────────────────────────────────
def _draw_overlay(frame, reps, phase, knee_angle, cues, last_feedback):
    """Burn a translucent HUD into *frame* (mutates in-place)."""
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (8, 8), (min(380, w - 8), 170), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Reps: {reps}", (18, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 80), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Phase: {phase}", (18, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Knee: {knee_angle:.0f} deg", (18, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)

    y = 140
    for c in cues:
        clr = (0, 255, 80) if "Good" in c else (0, 180, 255)
        cv2.putText(frame, c, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2, cv2.LINE_AA)
        y += 24

    if last_feedback:
        cv2.putText(frame, last_feedback, (18, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 255), 2, cv2.LINE_AA)
    return frame


def _show_summary(analyzer):
    """Render session summary + per-rep table below the video area."""
    summary = analyzer.get_session_summary()
    if summary["total_reps"] == 0:
        st.info("No reps were detected. Try adjusting your camera angle so your full body is visible.")
        return

    st.markdown("---")
    st.subheader("Session Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Reps", summary["total_reps"])
    c2.metric("Good", summary["good_reps"])
    c3.metric("Fair", summary["fair_reps"])
    c4.metric("Poor", summary["poor_reps"])
    c5.metric("Avg Score", summary["avg_score"])

    scores = analyzer.get_rep_scores()
    if scores:
        st.markdown("#### Per-Rep Breakdown")
        rows = [
            {
                "Rep": s.rep_number,
                "Quality": s.quality,
                "Overall": round(s.overall_score, 1),
                "Depth": round(s.depth_score, 1),
                "Knee": round(s.knee_score, 1),
                "Torso": round(s.torso_score, 1),
                "Lockout": round(s.lockout_score, 1),
                "Issues": ", ".join(i.value for i in s.issues) or "none",
            }
            for s in scores
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    if summary.get("issue_counts"):
        st.markdown("#### Common Issues")
        for issue, count in summary["issue_counts"].items():
            label = issue.replace("_", " ").title()
            st.write(f"- **{label}** -- {count} rep(s)")


def _update_sidebar(analyzer):
    """Push current metrics into the sidebar placeholders."""
    sb_reps.metric("Reps", analyzer.get_rep_count())
    sb_phase.text(f"Phase:  {analyzer.get_phase().value}")
    sb_angle.text(f"Knee angle:  {analyzer.get_knee_angle():.0f}\u00b0")

    cues = analyzer.get_feedback()
    sb_cues.markdown(
        "  \n".join(
            f"{'**' if 'Good' not in c else ''}{c}{'**' if 'Good' not in c else ''}"
            for c in cues
        )
    )

    last = analyzer.get_last_rep_feedback()
    if last:
        sb_last.success(last) if "Great" in last else sb_last.warning(last)


# ─── video-upload flow ───────────────────────────────────────────────
def run_video(path: str):
    from squat_form_checker import PoseEstimator, SquatAnalyzer

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.error("Could not open video file. The format may not be supported by your system's codecs.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.caption(f"Video: {w}x{h} @ {fps:.0f} fps  |  ~{total_frames} frames  |  ~{total_frames/fps:.1f}s")

    pe = PoseEstimator()
    analyzer = SquatAnalyzer(pe)

    frame_holder = st.empty()
    progress = st.progress(0)
    prog_status = st.empty()
    prog_status.caption("Analysing video...")
    frame_idx = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if ts_ms <= 0:
            ts_ms = int(frame_idx * 1000 / fps)

        landmarks, results = pe.process_frame(frame, timestamp_ms=max(ts_ms, frame_idx + 1))
        analyzer.process_landmarks(landmarks)
        annotated = pe.draw_landmarks(frame, results)

        reps = analyzer.get_rep_count()
        phase = analyzer.get_phase().value
        knee = analyzer.get_knee_angle()
        cues = analyzer.get_feedback()
        last = analyzer.get_last_rep_feedback()

        annotated = _draw_overlay(annotated, reps, phase, knee, cues, last)

        if frame_idx % 2 == 0:
            frame_holder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            _update_sidebar(analyzer)

        frame_idx += 1
        progress.progress(min(frame_idx / total_frames, 1.0))
        prog_status.caption(f"Frame {frame_idx}/{total_frames}")

    cap.release()
    pe.close()
    prog_status.empty()
    progress.empty()

    _show_summary(analyzer)


# ─── sidebar + main ─────────────────────────────────────────────────
if _deps_ok:
    with st.sidebar:
        st.markdown("#### Upload a squat video")
        st.caption("Supports iPhone (.MOV), Android (.mp4), and other common formats.")
        uploaded_file = st.file_uploader(
            "Drop your file here",
            type=["mp4", "mov", "avi", "mkv", "m4v", "webm"],
            key="squat_video_upload",
        )

        st.markdown("---")
        st.markdown("## Live Metrics")
        sb_reps = st.empty()
        sb_phase = st.empty()
        sb_angle = st.empty()
        st.markdown("---")
        st.markdown("## Coaching")
        sb_cues = st.empty()
        sb_last = st.empty()

    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        run_video(tmp_path)
    else:
        st.markdown(
            """
            ### How to use
            1. Record a squat video on your **phone** (iPhone or Android).
            2. Use the **upload button in the sidebar** to select your video.
            3. The app will analyse every frame and give you a rep-by-rep breakdown.

            **Tips for best results**
            - Film from the **side** for the most accurate depth and torso-lean detection.
            - Make sure your **full body** (head to feet) is visible in the frame.
            - Good lighting helps the pose model track you reliably.
            """,
        )
