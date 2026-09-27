import glob
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

# ---- MUHIM: haqiqiy model tayyor bo'lganda shu qatorni almashtiring ----
from solution_stub import detect_events, RiskEstimator, CLASSES
# from solution import detect_events, RiskEstimator, CLASSES
# --------------------------------------------------------------------

st.set_page_config(
    page_title="Traffic Event Detection — NOWL Hackathon",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Dizayn tokenlari
# ---------------------------------------------------------------------------

BG = "#0A0D12"
PANEL = "#12161D"
PANEL_2 = "#161B24"
BORDER = "#232935"
TEXT = "#E7EAEE"
MUTED = "#8B94A3"
ACCENT = "#F5A524"      # amber — svetofor rangi
ACCENT_2 = "#4C8DFF"    # ko'k — ikkilamchi
DANGER = "#EF4444"
SUCCESS = "#22C55E"

# 14 rasmiy sinf uchun barqaror, qasddan tanlangan rang spektri
CLASS_PALETTE = [
    "#EF4444", "#F97316", "#F5A524", "#EAB308", "#84CC16",
    "#22C55E", "#14B8A6", "#06B6D4", "#4C8DFF", "#6366F1",
    "#8B5CF6", "#D946EF", "#EC4899", "#F43F5E",
]
CLASS_COLORS = {c: CLASS_PALETTE[i % len(CLASS_PALETTE)] for i, c in enumerate(CLASSES)}

PLOTLY_FONT = dict(family="Inter, sans-serif", color=TEXT)


def style_fig(fig, height=None):
    fig.update_layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=PLOTLY_FONT,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    if height:
        fig.update_layout(height=height)
    return fig


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
h1, h2, h3, h4 {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}}
code, .mono {{
    font-family: 'JetBrains Mono', monospace !important;
}}

.stApp {{
    background: {BG};
    color: {TEXT};
}}

/* Tabs */
[data-testid="stTabs"] button {{
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    color: {MUTED};
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {TEXT};
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    background-color: {ACCENT} !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{
    background-color: {BORDER};
}}

/* Bordered containers -> cards */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {PANEL};
    border: 1px solid {BORDER} !important;
    border-radius: 10px;
}}

/* Metrics */
[data-testid="stMetric"] {{
    background: {PANEL_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 12px 16px;
}}
[data-testid="stMetricLabel"] {{
    color: {MUTED};
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace;
    color: {TEXT};
}}

/* Alerts */
.stAlert {{
    border-radius: 8px;
    border: 1px solid {BORDER};
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    overflow: hidden;
}}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {{
    background: {PANEL_2};
    border: 1px dashed {BORDER};
    border-radius: 8px;
}}

/* Buttons / select */
.stSelectbox [data-baseweb="select"] {{
    border-radius: 8px;
}}

/* Divider */
hr {{
    border-color: {BORDER};
}}

.pill {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    background: rgba(245, 165, 36, 0.12);
    color: {ACCENT};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    border: 1px solid rgba(245, 165, 36, 0.35);
}}
.pill-muted {{
    background: rgba(139, 148, 163, 0.12);
    color: {MUTED};
    border-color: {BORDER};
}}
.hero-title {{
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 2px;
}}
.hero-sub {{
    color: {MUTED};
    font-size: 0.95rem;
    margin-bottom: 0;
}}
</style>
""", unsafe_allow_html=True)


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
        return style_fig(fig, height=120)
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
    return style_fig(fig)


def risk_figure(ts, scores):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=scores, mode="lines", name="Risk",
        line=dict(color=ACCENT_2, width=2),
        fill="tozeroy", fillcolor="rgba(76,141,255,0.08)",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color=DANGER,
                  annotation_text="θ = 0.5 (alarm chegarasi)",
                  annotation_font_color=DANGER)
    fig.update_layout(xaxis_title="Vaqt (s)", yaxis_title="P(avariya ≤ 5s ichida)",
                       yaxis_range=[0, 1])
    return style_fig(fig, height=320)


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
# Header
# ---------------------------------------------------------------------------

head_l, head_r = st.columns([4, 1])
with head_l:
    st.markdown('<div class="hero-title">Traffic Event Detection &amp; Accident Anticipation</div>',
                unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Fixed road camera · event detection and accident anticipation</p>',
                unsafe_allow_html=True)
with head_r:
    st.markdown(
        '<div style="text-align:right; padding-top:14px;">'
        '<span class="pill">NOWL · E2A46D42</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

tabs = st.tabs(["Jamoa", "Yondashuv", "EDA", "Live Demo", "Natijalar", "Hisobot"])

# ---- Jamoa ----
with tabs[0]:
    st.header("Jamoa")

    TEAM = [
        {
            "name": "Anvar Mexmonov",
            "role": "AI & Pipeline Engineer",
            "task": "YOLO/ByteTrack integratsiyasi, Part A (hodisa qoidalari) va Part B (RiskEstimator) mantiqi",
            "github": "https://github.com/anvarmexmonov",
            "linkedin": "https://www.linkedin.com/in/anvar-mexmonov",
            "prev_project": "",
        },
        {
            "name": "Sirojiddin Abduraxmonov",
            "role": "Data, Evaluation & Quality Engineer",
            "task": "DevSet (ground_truth_dev.json) tayyorlash, evaluate.py orqali sifat nazorati, vaqt/resurs cheklovlariga moslik, repo/README",
            "github": "https://github.com/sityuz",
            "linkedin": "https://www.linkedin.com/in/sirojiddin-abduraxmonov-81474a363",
            "prev_project": "",
        },
        {
            "name": "Sardor Omonov",
            "role": "Full-Stack Web & Demo Engineer",
            "task": "Jamoa websayti, EDA vizualizatsiyasi, Live Demo va Natijalar sahifalari",
            "github": "https://github.com/omonovs",
            "linkedin": "https://www.linkedin.com/in/sardor-omonov-b5139338a",
            "prev_project": "",
        },
    ]

    cols = st.columns(3)
    for col, member in zip(cols, TEAM):
        with col:
            with st.container(border=True):
                st.markdown(f"**{member['name']}**")
                st.markdown(f'<span class="pill pill-muted">{member["role"]}</span>',
                            unsafe_allow_html=True)
                st.markdown(f"<div style='color:{MUTED}; font-size:0.9rem; margin-top:8px;'>"
                            f"{member['task']}</div>", unsafe_allow_html=True)
                links = []
                if member.get("github"):
                    links.append(f"[GitHub]({member['github']})")
                if member.get("linkedin"):
                    links.append(f"[LinkedIn]({member['linkedin']})")
                if links:
                    st.markdown(" · ".join(links))
                if member.get("prev_project"):
                    st.caption(f"Oldingi loyiha: {member['prev_project']}")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Jamoaviy tajriba**")
        st.write(
            "Ilgari CAU (Central Asian University) o'tkazgan Healthcare hackathonida "
            "ishtirokchi bo'lganmiz, shuningdek TUIT ichki hackathonlarida 2 marta "
            "qatnashganmiz (g'olib bo'lmagan bo'lsak ham, tajriba to'plaganmiz)."
        )

    st.caption("Jamoa nomi: NOWL TEAM ·  Computer Vision Hackathon · E2A46D42")

# ---- Yondashuv ----
with tabs[1]:
    st.header("Muammo va yondashuv")

    with st.container(border=True):
        st.subheader("Pipeline")
        st.markdown(
            "`Video` → `Frame extraction` → `Object detection (YOLO)` → "
            "`Tracking (ByteTrack)` → `Rule-based event logic (Part A)` → "
            "`Risk scoring (Part B)` → `predictions.json`"
        )

    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.markdown("**Qoidaga asoslangan (rule-based)**")
            st.markdown(
                "- `stopped_vehicle` — 10s harakatsizlik\n"
                "- `wrong_way` — harakat vektori tahlili\n"
                "- `jaywalking` — piyoda / yo'l zonasi kesishishi\n"
                "- `congestion` — tezlik pasayishi + zichlik"
            )
    with col_b:
        with st.container(border=True):
            st.markdown("**O'rganilgan (learned)**")
            st.markdown(
                "- Obyekt detektsiya — YOLOv8x / RT-DETR\n"
                "- Tracking — ByteTrack"
            )

    st.warning(
        "Diagramma rasmi va ishlatilgan datasetlar/litsenziyalar hali qo'shilmagan — "
        "1 va 2-ishtirokchi bilan kelishib to'ldiriladi."
    )

# ---- EDA ----
with tabs[2]:
    st.header("Sample videolar bo'yicha EDA")

    eda_json_path = os.path.join("assets", "eda_stats.json")
    if os.path.exists(eda_json_path):
        with open(eda_json_path, "r", encoding="utf-8") as f:
            precomputed = json.load(f)

        with st.container(border=True):
            st.subheader("Sample videolar statistikasi")
            pre_df = pd.DataFrame(precomputed)[
                ["name", "fps", "width", "height", "n_frames", "duration_sec"]
            ]
            st.dataframe(pre_df, width="stretch")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Harakat issiqlik xaritalari")
        cols = st.columns(2)
        for i, row in enumerate(precomputed):
            hm = row.get("heatmap_file")
            if hm:
                hm_path = os.path.join("assets", "heatmaps", hm)
                if os.path.exists(hm_path):
                    with cols[i % 2]:
                        with st.container(border=True):
                            st.image(hm_path, caption=row["name"], width="stretch")
        st.divider()
    else:
        st.info(
            "Statik EDA natijalari hali generatsiya qilinmagan. "
            "`python precompute_eda.py` skriptini ishga tushiring."
        )

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

            with st.expander(f"{f.name} — harakat issiqlik xaritasi"):
                heat = motion_heatmap(tmp_path)
                if heat is not None:
                    fig = px.imshow(heat, color_continuous_scale="inferno",
                                     labels=dict(color="Harakat intensivligi"))
                    style_fig(fig, height=300)
                    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.write("Videoda kadr topilmadi.")
            os.unlink(tmp_path)

        df = pd.DataFrame(rows)[["name", "fps", "width", "height", "n_frames", "duration_sec"]]
        st.subheader("Video statistikasi")
        st.dataframe(df, width="stretch")
    else:
        st.info("EDA uchun bir nechta sample videoni shu yerga yuklang.")

    with st.container(border=True):
        st.markdown("**Keyingi topilmalar (rejalashtirilgan)**")
        st.markdown(
            "- Yorug'lik sharoiti (kun/tun, soya)\n"
            "- Vaqt bo'yicha obyektlar soni (mashina/piyoda) grafigi\n"
            "- Yo'l yo'nalishlari va traektoriyalar xaritasi\n"
            "- Traffic zichligi vaqt bo'yicha"
        )

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
        progress.info("Hodisalar aniqlanmoqda (Part A)...")
        t0 = time.time()
        events = detect_events(video_path)
        t1 = time.time()
        progress.success(f"{len(events)} ta hodisa topildi ({t1 - t0:.1f}s)")

        stats = video_basic_stats(video_path)
        duration = stats["duration_sec"]

        st.subheader("Hodisalar jadvali")
        if events:
            ev_df = pd.DataFrame(events, columns=["Boshlanish (s)", "Tugash (s)", "Sinf"])
            st.dataframe(ev_df, width="stretch")
        else:
            st.write("Hodisa topilmadi.")

        st.subheader("Vaqt chizig'i (timeline)")
        st.plotly_chart(timeline_figure(events, duration), width="stretch")

        st.subheader("Xavf grafigi (Risk curve, Part B)")
        with st.spinner("Risk hisoblanmoqda..."):
            ts, scores = risk_curve(video_path)
        st.plotly_chart(risk_figure(ts, scores), width="stretch")

        os.unlink(video_path)
    else:
        st.info("Demo uchun video yuklang.")

# ---- Natijalar ----
with tabs[4]:
    st.header("Sample videolar bo'yicha natijalar")
    st.caption(
        "Har bir sample video oldindan precompute_results.py skripti bilan "
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
            "Quyidagi natijalar hozircha stub model bilan olingan. "
            "1-ishtirokchining haqiqiy solution.py fayli tayyor bo'lgach, "
            "precompute_results.py qayta ishga tushiriladi va bu sahifa "
            "avtomatik yangi (haqiqiy) natijalarni ko'rsatadi."
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
            with st.container(border=True):
                annotated_name = data.get("annotated_video")
                annotated_path = os.path.join(RESULTS_DIR, annotated_name) if annotated_name else None
                if annotated_path and os.path.exists(annotated_path):
                    st.video(annotated_path)
                else:
                    st.warning("Annotatsiyalangan video topilmadi.")

        with col_right:
            m1, m2 = st.columns(2)
            m1.metric("Davomiyligi", f"{data['duration_sec']} s")
            m2.metric("FPS", data["fps"])
            m3, m4 = st.columns(2)
            m3.metric("Topilgan hodisalar", len(data["events"]))
            max_risk = max((r[1] for r in data["risk"]), default=0.0)
            m4.metric("Eng yuqori risk", f"{max_risk:.2f}")

        st.subheader("Hodisalar jadvali")
        if data["events"]:
            ev_df = pd.DataFrame(data["events"], columns=["Boshlanish (s)", "Tugash (s)", "Sinf"])
            st.dataframe(ev_df, width="stretch")
        else:
            st.write("Hodisa topilmadi.")

        st.subheader("Vaqt chizig'i (timeline)")
        st.plotly_chart(timeline_figure(data["events"], data["duration_sec"]), width="stretch")

        st.subheader("Xavf grafigi (Risk curve, Part B)")
        ts = [r[0] for r in data["risk"]]
        scores = [r[1] for r in data["risk"]]
        st.plotly_chart(risk_figure(ts, scores), width="stretch")

        with st.expander("Halol failure case / cheklovlar"):
            st.markdown(
                "- Hozircha stub model — hodisalar tasodifiy generatsiya qilingan, "
                "haqiqiy detektsiya emas.\n"
                "- Haqiqiy model ulanganda bu yerga: model xato qilgan aniq holatlar "
                "(masalan noto'g'ri sinf, o'tkazib yuborilgan hodisa) yoziladi."
            )

        st.caption(f"Barcha {len(result_files)} ta sample video uchun natijalar tayyor.")

# ---- Hisobot ----
with tabs[5]:
    st.header("Texnik hisobot")
    st.caption("Bir sahifalik xulosa — nima ishladi, nima ishlamadi, keyingi qadamlar.")

    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.markdown("**Nima ishladi**")
            st.markdown("- TODO")
    with col_b:
        with st.container(border=True):
            st.markdown("**Nima ishlamadi / qiyinchiliklar**")
            st.markdown("- TODO")

    with st.container(border=True):
        st.markdown("**Keyingi qadamlar (agar davom etsa)**")
        st.markdown("- TODO")

    with st.container(border=True):
        st.markdown("**Havolalar**")
        st.markdown(
            "- Repository: TODO\n"
            "- Weights: TODO\n"
            "- predictions_samples.json: TODO"
        )

st.divider()
st.caption(
    "Bu sahifa hozircha stub model (solution_stub.py) bilan ishlayapti — "
    "1-ishtirokchining haqiqiy solution.py tayyor bo'lgach, app.py dagi import "
    "qatorini almashtiring."
)