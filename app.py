import json
import os
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import glob

# ---- MUHIM: haqiqiy model tayyor bo'lganda shu qatorni almashtiring ----
from solution_stub import detect_events, RiskEstimator, CLASSES
# from solution import detect_events, RiskEstimator, CLASSES
# --------------------------------------------------------------------

st.set_page_config(page_title="Traffic Event Detection — NOWL Hackathon", layout="wide")

CLASS_COLORS = {c: px.colors.qualitative.Dark24[i % 24] for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------------------------
# Yordamchi funksiyalar
# ---------------------------------------------------------------------------

def video_basic_stats(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / fps if fps else 0
    cap.release()
    return {"fps": round(fps, 2), "width": w, "height": h,
            "n_frames": n, "duration_sec": round(dur, 1)}


def motion_heatmap(path: str, max_samples: int = 120):
    """Kadrlar farqi (frame-diff) orqali harakat issiqlik xaritasi."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, n // max_samples)
    prev = None
    acc = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))
            if prev is not None:
                diff = cv2.absdiff(gray, prev)
                acc = diff.astype(np.float32) if acc is None else acc + diff.astype(np.float32)
            prev = gray
        idx += 1
    cap.release()
    if acc is None:
        return None
    acc = acc / (acc.max() + 1e-6)
    return acc


def timeline_figure(events: list, duration: float):
    if not events:
        fig = go.Figure()
        fig.update_layout(title="Hodisalar topilmadi", xaxis_title="Vaqt (s)")
        return fig
    df = pd.DataFrame(events, columns=["start", "end", "label"])
    fig = px.timeline(
        df.assign(start_dt=pd.to_datetime(df.start, unit="s"),
                  end_dt=pd.to_datetime(df.end, unit="s")),
        x_start="start_dt", x_end="end_dt", y="label", color="label",
        color_discrete_map=CLASS_COLORS,
    )
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title="Vaqt")
    fig.update_layout(showlegend=False, height=80 + 35 * df["label"].nunique())
    return fig


def risk_curve(path: str, sample_every_n: int = 5, max_points: int = 400):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    est = RiskEstimator()
    est.reset({"video_id": os.path.basename(path), "fps": fps,
               "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "n_frames": n})
    ts, scores = [], []
    idx = 0
    step = max(1, (n // max_points) if n else sample_every_n)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t = idx / fps
            s = est.step(frame, t)
            ts.append(t)
            scores.append(s)
        idx += 1
    cap.release()
    return ts, scores


# ---------------------------------------------------------------------------
# Sidebar navigatsiya
# ---------------------------------------------------------------------------

st.title("🚦 Traffic Event Detection & Accident Anticipation")
st.caption("NOWL Computer Vision Hackathon · E2A46D42")

tabs = st.tabs(["👥 Jamoa", "🧭 Yondashuv", "📊 EDA", "🎬 Live Demo", "📁 Natijalar", "📝 Hisobot"])

# ---- Jamoa ----
with tabs[0]:
    st.header("Jamoa")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Ism Familiya")
        st.write("**Rol:** AI & Pipeline Engineer")
        st.write("Vazifa: YOLO/ByteTrack, Part A & B mantiqi")
        st.markdown("[GitHub](#) · [LinkedIn](#)")
    with col2:
        st.subheader("Ism Familiya")
        st.write("**Rol:** Data, Evaluation & Quality Engineer")
        st.write("Vazifa: Dev-set, evaluate.py, resurslar nazorati, README")
        st.markdown("[GitHub](#) · [LinkedIn](#)")
    with col3:
        st.subheader("Sardor")
        st.write("**Rol:** Full-Stack Web & Demo Engineer")
        st.write("Vazifa: Sayt, EDA vizualizatsiyasi, Live Demo")
        st.markdown("[GitHub](#) · [LinkedIn](#)")
    st.info("TODO: Har bir a'zoning ismi, real rasmi, portfolio va oldingi loyihalarini qo'shing.")

# ---- Yondashuv ----
with tabs[1]:
    st.header("Muammo va Yondashuv")
    st.markdown("""
**Pipeline (umumiy chizma):**

`Video → Frame extraction → Object detection (YOLO) → Tracking (ByteTrack) →
Rule-based event logic (Part A) → Risk scoring (Part B) → predictions.json`

- **Rule-based qismlar:** stopped_vehicle (10s harakatsizlik), wrong_way
  (harakat vektori tahlili), jaywalking (piyoda-yo'l zonasi kesishishi),
  congestion (tezlik pasayishi + zichlik).
- **Learned qismlar:** obyekt detektsiya (YOLOv8x/RT-DETR), tracking (ByteTrack).

TODO: Chizmani rasm/diagram sifatida qo'shing (draw.io yoki excalidraw bilan
chizib, screenshot qilib joylashtirsangiz bo'ladi). Ishlatilgan datasetlar va
litsenziyalarini shu yerga yozing.
    """)
    st.warning("Bu bo'lim TODO — 1 va 2-ishtirokchi bilan kelishib to'ldiring.")

# ---- EDA ----
with tabs[2]:
    st.header("Sample videolar bo'yicha EDA")

    # --- Oldindan hisoblangan (statik) natijalar — hech narsa yuklamasdan ko'rinadi ---
    eda_json_path = os.path.join("assets", "eda_stats.json")
    if os.path.exists(eda_json_path):
        with open(eda_json_path, "r", encoding="utf-8") as f:
            precomputed = json.load(f)
        st.subheader("Sample videolar statistikasi")
        pre_df = pd.DataFrame(precomputed)[
            ["name", "fps", "width", "height", "n_frames", "duration_sec"]
        ]
        st.dataframe(pre_df, use_container_width=True)

        st.subheader("Harakat issiqlik xaritalari")
        cols = st.columns(2)
        for i, row in enumerate(precomputed):
            hm = row.get("heatmap_file")
            if hm:
                hm_path = os.path.join("assets", "heatmaps", hm)
                if os.path.exists(hm_path):
                    with cols[i % 2]:
                        st.image(hm_path, caption=row["name"], use_container_width=True)
        st.divider()
    else:
        st.info(
            "Statik EDA natijalari hali generatsiya qilinmagan. "
            "`python precompute_eda.py` skriptini ishga tushiring."
        )

    # --- Interaktiv qism: tashrif buyuruvchi o'z videosini yuklab ko'rishi mumkin ---
    st.subheader("O'zingiz sinab ko'ring")
    uploaded = st.file_uploader(
        "Istalgan video yuklang (.mp4) — tahlil shu yerda ko'rsatiladi",
        type=["mp4"], accept_multiple_files=True, key="eda_upload",
    )
    if uploaded:
        rows = []
        for f in uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            stats = video_basic_stats(tmp_path)
            stats["name"] = f.name
            rows.append(stats)

            with st.expander(f"🎞 {f.name} — harakat issiqlik xaritasi"):
                heat = motion_heatmap(tmp_path)
                if heat is not None:
                    fig = px.imshow(heat, color_continuous_scale="inferno",
                                     labels=dict(color="Harakat intensivligi"))
                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Videoda kadr topilmadi.")
            os.unlink(tmp_path)

        df = pd.DataFrame(rows)[["name", "fps", "width", "height", "n_frames", "duration_sec"]]
        st.subheader("Video statistikasi")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("EDA uchun bir nechta sample videoni shu yerga yuklang.")

    st.markdown("""
**TODO (topilmalar shu yerga yoziladi):**
- Yorug'lik sharoiti (kun/tun, soya)
- Vaqt bo'yicha obyektlar soni (mashina/piyoda) grafigi
- Yo'l yo'nalishlari va traektoriyalar xaritasi
- Traffic zichligi vaqt bo'yicha
    """)

# ---- Live Demo ----
with tabs[3]:
    st.header("Live Demo")
    st.caption("Video yuklang (≤ 2 daqiqa tavsiya etiladi) — model hodisalarni va xavf darajasini aniqlaydi.")
    demo_file = st.file_uploader("Video (.mp4)", type=["mp4"], key="demo_upload")

    if demo_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(demo_file.read())
            video_path = tmp.name

        st.video(demo_file)

        progress = st.empty()
        progress.info("⏳ Hodisalar aniqlanmoqda (Part A)...")
        t0 = time.time()
        events = detect_events(video_path)
        t1 = time.time()
        progress.success(f"✅ {len(events)} ta hodisa topildi ({t1 - t0:.1f}s)")

        stats = video_basic_stats(video_path)
        duration = stats["duration_sec"]

        st.subheader("Hodisalar jadvali")
        if events:
            ev_df = pd.DataFrame(events, columns=["Boshlanish (s)", "Tugash (s)", "Sinf"])
            st.dataframe(ev_df, use_container_width=True)
        else:
            st.write("Hodisa topilmadi.")

        st.subheader("Vaqt chizig'i (timeline)")
        st.plotly_chart(timeline_figure(events, duration), use_container_width=True)

        st.subheader("Xavf grafigi (Risk curve, Part B)")
        with st.spinner("Risk hisoblanmoqda..."):
            ts, scores = risk_curve(video_path)
        risk_fig = go.Figure()
        risk_fig.add_trace(go.Scatter(x=ts, y=scores, mode="lines", name="Risk"))
        risk_fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                            annotation_text="θ = 0.5 (alarm chegarasi)")
        risk_fig.update_layout(xaxis_title="Vaqt (s)", yaxis_title="P(avariya ≤ 5s ichida)",
                                yaxis_range=[0, 1], height=350)
        st.plotly_chart(risk_fig, use_container_width=True)

        os.unlink(video_path)
    else:
        st.info("Demo uchun video yuklang.")

# ---- Natijalar ----
# ============================================================================
# APP.PY GA O'ZGARTIRISH QO'LLANMASI
#
# 1) app.py boshidagi importlar qatoriga (boshqa importlardan keyin) qo'shing:
#      import glob
#
# 2) app.py dagi eski "Natijalar" blokini (with tabs[4]: ... st.info("""TODO...""")
#    qismini) TO'LIQ o'chirib, o'rniga pastdagi blokni qo'ying.
# ============================================================================

with tabs[4]:
    st.header("Sample videolar bo'yicha natijalar")
    st.caption(
        "Har bir sample video oldindan `precompute_results.py` skripti bilan "
        "qayta ishlangan — hakamlar hech narsa yuklamasdan natijalarni ko'radi."
    )

    RESULTS_DIR = os.path.join("assets", "results")
    result_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))

    if not result_files:
        st.warning(
            "Natijalar hali generatsiya qilinmagan. Terminalda quyidagini ishga "
            "tushiring:\n\n`python precompute_results.py`\n\nSo'ng `assets/results/` "
            "papkasini va `predictions_samples.json` faylini push qiling."
        )
    else:
        st.info(
            "⚠️ Quyidagi natijalar hozircha **stub model** bilan olingan. "
            "1-ishtirokchining haqiqiy `solution.py` fayli tayyor bo'lgach, "
            "`precompute_results.py` qayta ishga tushiriladi va bu sahifa "
            "avtomatik yangi (haqiqiy) natijalarni ko'rsatadi.",
            icon="⚠️",
        )

        video_names = []
        for fp in result_files:
            with open(fp, "r", encoding="utf-8") as f:
                video_names.append((fp, json.load(f)))

        pick = st.selectbox(
            "Sample videoni tanlang",
            options=list(range(len(video_names))),
            format_func=lambda i: video_names[i][1]["name"],
        )
        fp, data = video_names[pick]

        col_left, col_right = st.columns([3, 2])

        with col_left:
            annotated_name = data.get("annotated_video")
            annotated_path = os.path.join(RESULTS_DIR, annotated_name) if annotated_name else None
            if annotated_path and os.path.exists(annotated_path):
                st.video(annotated_path)
            else:
                st.warning("Annotatsiyalangan video topilmadi.")

        with col_right:
            st.metric("Davomiyligi", f"{data['duration_sec']} s")
            st.metric("FPS", data["fps"])
            st.metric("Topilgan hodisalar", len(data["events"]))
            max_risk = max((r[1] for r in data["risk"]), default=0.0)
            st.metric("Eng yuqori risk", f"{max_risk:.2f}")

        st.subheader("Hodisalar jadvali")
        if data["events"]:
            ev_df = pd.DataFrame(data["events"], columns=["Boshlanish (s)", "Tugash (s)", "Sinf"])
            st.dataframe(ev_df, use_container_width=True)
        else:
            st.write("Hodisa topilmadi.")

        st.subheader("Vaqt chizig'i (timeline)")
        st.plotly_chart(timeline_figure(data["events"], data["duration_sec"]), use_container_width=True)

        st.subheader("Xavf grafigi (Risk curve, Part B)")
        ts = [r[0] for r in data["risk"]]
        scores = [r[1] for r in data["risk"]]
        risk_fig = go.Figure()
        risk_fig.add_trace(go.Scatter(x=ts, y=scores, mode="lines", name="Risk"))
        risk_fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                            annotation_text="θ = 0.5 (alarm chegarasi)")
        risk_fig.update_layout(xaxis_title="Vaqt (s)", yaxis_title="P(avariya ≤ 5s ichida)",
                                yaxis_range=[0, 1], height=350)
        st.plotly_chart(risk_fig, use_container_width=True)

        with st.expander("📉 Halol failure case / cheklovlar"):
            st.markdown(
                "- Hozircha stub model — hodisalar tasodifiy generatsiya qilingan, "
                "haqiqiy detektsiya emas.\n"
                "- Haqiqiy model ulanganda bu yerga: model xato qilgan aniq holatlar "
                "(masalan noto'g'ri sinf, o'tkazib yuborilgan hodisa) yoziladi."
            )

        st.divider()
        st.caption(f"Barcha {len(result_files)} ta sample video uchun natijalar tayyor.")

# ---- Hisobot ----
with tabs[5]:
    st.header("Texnik hisobot (1 sahifa)")
    st.markdown("""
**Nima ishladi:**
- TODO

**Nima ishlamadi / qiyinchiliklar:**
- TODO

**Keyingi qadamlar (agar davom etsa):**
- TODO

**Havolalar:**
- Repository: TODO
- Weights: TODO
- predictions_samples.json: TODO
    """)

st.divider()
st.caption("⚠️ Bu sahifa hozircha STUB model (`solution_stub.py`) bilan ishlayapti — "
           "1-ishtirokchining haqiqiy solution.py tayyor bo'lgach, app.py dagi import "
           "qatorini almashtiring.")