import os
import json
import tempfile
from datetime import datetime

import streamlit as st
from PIL import Image

from pipeline import run_pipeline

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BeltGuard",
    page_icon="B",
    layout="wide"
)


# ============================================================
# PROFESSIONAL UI THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --ink: #0b1220;
        --muted: #64748b;
        --line: #e2e8f0;
        --panel: #ffffff;
        --bg: #f4f7fa;
        --navy: #0b1728;
        --navy-2: #10243a;
        --accent: #0f766e;
        --accent-2: #14b8a6;
    }

    html, body, .stApp {
        font-family: "Segoe UI", "Inter", Arial, sans-serif !important;
    }

    .stApp { background: var(--bg); }
    .main .block-container {
        max-width: 1480px;
        padding-top: 1.15rem;
        padding-bottom: 3.5rem;
    }

    /* Industrial control-room sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081321 0%, #0d1b2d 100%);
        border-right: 1px solid #1f334b;
    }
    section[data-testid="stSidebar"] * { color: #dce7f3 !important; }
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #aebfd2 !important;
        font-size: .76rem !important;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] .stSlider > div > div > div { background: #14b8a6; }

    /* Main typography */
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] strong,
    section[data-testid="stMain"] label,
    section[data-testid="stMain"] label p,
    section[data-testid="stMain"] [data-testid="stWidgetLabel"] p { color: #233044 !important; }
    section[data-testid="stMain"] h1, section[data-testid="stMain"] h2,
    section[data-testid="stMain"] h3, section[data-testid="stMain"] h4 { color: var(--ink) !important; }
    section[data-testid="stMain"] .section-title { 
        font-size: 1.12rem !important;
        line-height: 1.35 !important;
    }
    section[data-testid="stMain"] .section-subtitle {
        font-size: 0.84rem !important;
        line-height: 1.5 !important;
        color: #66788c !important;
    }
    section[data-testid="stMain"] [data-testid="stCaptionContainer"],
    section[data-testid="stMain"] [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

    /* Hero / product masthead */
    .belt-hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(112deg, #071321 0%, #0c2137 62%, #0e4f58 100%);
        padding: 2.15rem 2.15rem 2.05rem;
        border: 1px solid #17334a;
        border-radius: 12px;
        margin-bottom: 1.15rem;
        box-shadow: 0 14px 34px rgba(8, 19, 33, .12);
        color: white;
    }
    .belt-hero:after {
        content: ""; position: absolute; width: 240px; height: 240px;
        right: -70px; top: -105px; border-radius: 50%;
        border: 1px solid rgba(45,212,191,.16);
        box-shadow: 0 0 0 32px rgba(45,212,191,.035), 0 0 0 64px rgba(45,212,191,.025);
    }
    .belt-brandline { display:flex; align-items:center; gap:.7rem; margin-bottom:.85rem; }
    .belt-kicker { color:#78a7bd; font-size:.68rem; letter-spacing:.14em; font-weight:800; text-transform:uppercase; }
    .belt-hero h1 { margin:0 !important; font-size:2.65rem; font-weight:800; letter-spacing:-.035em; color:#f8fafc !important; -webkit-text-fill-color:#f8fafc !important; text-shadow:none !important; }
    .belt-title-rule { width:58px; height:3px; background:#2dd4bf; border-radius:3px; margin:.72rem 0 .68rem; }
    .belt-hero p { margin:.0rem 0 0; max-width:820px; color:#d8e7ef !important; -webkit-text-fill-color:#d8e7ef !important; font-size:1.02rem; font-weight:500; line-height:1.6; letter-spacing:.005em; }
    .belt-hero h1, .belt-hero h1 span, .belt-hero [data-testid="stMarkdownContainer"] h1 { color:#f8fafc !important; -webkit-text-fill-color:#f8fafc !important; }
    .belt-live {
        position:absolute; right:1.35rem; bottom:1.25rem; z-index:2;
        display:flex; align-items:center; gap:.45rem; color:#b7f7ef;
        font-size:.68rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase;
    }
    .belt-dot { width:7px; height:7px; border-radius:50%; background:#2dd4bf; box-shadow:0 0 0 4px rgba(45,212,191,.12); }

    .ops-strip {
        display:grid; grid-template-columns:repeat(4,1fr); gap:0;
        background:#fff; border:1px solid #dbe3ea; border-radius:10px;
        overflow:hidden; margin:0 0 1.25rem; box-shadow:0 2px 8px rgba(15,23,42,.025);
    }
    .ops-strip > div { padding:.72rem .9rem; border-right:1px solid #e5ebf0; }
    .ops-strip > div:last-child { border-right:0; }
    .ops-label { display:block; color:#8a98a8; font-size:.62rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.22rem; }
    .ops-value { color:#253446; font-size:.76rem; font-weight:750; letter-spacing:.03em; }
    .ops-value.online { color:#0f766e; }

    .section-title { font-size:1.12rem; font-weight:750; color:var(--ink); margin:1.35rem 0 .28rem; letter-spacing:-.01em; }
    .section-subtitle { color:#66788c !important; font-size:.84rem !important; margin:0 0 .8rem; }

    /* Cards and metrics */
    div[data-testid="stMetric"] {
        background:#fff; border:1px solid var(--line); border-radius:10px;
        padding:.82rem .9rem; box-shadow:0 2px 8px rgba(15,23,42,.035);
    }
    div[data-testid="stMetricLabel"] { color:#718096 !important; font-weight:700; font-size:.72rem; text-transform:uppercase; letter-spacing:.055em; }
    div[data-testid="stMetricValue"] { color:#101827 !important; font-weight:800; font-size:1.45rem; }
    .info-card, .status-card {
        background:#fff; border:1px solid var(--line); border-radius:10px;
        padding:1rem 1.05rem; box-shadow:0 2px 8px rgba(15,23,42,.035);
    }
    .info-card { min-height: 138px; }
    /* Inputs: crisp white industrial controls */


    section[data-testid="stMain"] [data-baseweb="input"],
    section[data-testid="stMain"] [data-testid="stNumberInputContainer"] {
        background:#fff !important; border-radius:8px !important;
    }
    section[data-testid="stMain"] [data-baseweb="input"] input,
    section[data-testid="stMain"] input[type="number"],
    section[data-testid="stMain"] input[type="text"] {
        color:#101827 !important; -webkit-text-fill-color:#101827 !important;
        background:#fff !important; font-weight:600 !important;
    }
    section[data-testid="stMain"] [data-baseweb="input"] input::placeholder { color:#94a3b8 !important; }
    section[data-testid="stMain"] [data-testid="stNumberInputStepDown"],
    section[data-testid="stMain"] [data-testid="stNumberInputStepUp"] { color:#475569 !important; }

    /* Buttons */
    .stButton > button {
        min-height:2.55rem; border-radius:8px; font-weight:750;
        border:1px solid #cbd5e1; transition:.16s ease;
    }
    .stButton > button:hover { transform:translateY(-1px); box-shadow:0 5px 14px rgba(15,23,42,.10); }
    .stButton > button[kind="primary"] { background:#0f766e !important; color:#fff !important; border-color:#0f766e !important; }
    .stButton > button[kind="primary"] p, .stButton > button[kind="primary"] span { color:#fff !important; }

    /* Tables / alerts */
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:9px; overflow:hidden; }
    div[data-testid="stAlert"] { border-radius:9px; border-width:1px; }
    hr { border:0; border-top:1px solid var(--line); margin:1.25rem 0; }

    .result-banner {
        display:flex; align-items:center; justify-content:space-between; gap:1rem;
        background:#f8fafc; border:1px solid #dce5ee; border-radius:10px;
        padding:.78rem 1rem; margin:.65rem 0 1rem;
    }
    .result-banner .eyebrow { font-size:.65rem; text-transform:uppercase; letter-spacing:.1em; color:#718096; font-weight:800; }
    .result-banner .value { font-size:.95rem; color:#0f172a; font-weight:800; margin-top:.15rem; }
    .priority-chip { display:inline-flex; align-items:center; border-radius:999px; padding:.32rem .62rem; font-size:.66rem; font-weight:800; letter-spacing:.06em; background:#ecfdf5; color:#047857; border:1px solid #a7f3d0; }
    .footer-line { text-align:center; color:#94a3b8; font-size:.72rem; padding:1.4rem 0 .4rem; letter-spacing:.02em; }

    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}

    /* Keep Streamlit heading anchors from appearing like app icons */
    section[data-testid="stMain"] [data-testid="stHeaderActionElements"] { display:none !important; }
    section[data-testid="stMain"] a[href^="#"] { display:none !important; }

    /* Clean number inputs */
    section[data-testid="stMain"] [data-baseweb="input"] {
        background:#ffffff !important;
        border:1px solid #d8e1e8 !important;
        border-radius:8px !important;
    }
    section[data-testid="stMain"] [data-baseweb="input"] input {
        color:#142235 !important;
        -webkit-text-fill-color:#142235 !important;
        background:#ffffff !important;
        font-size:.92rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATUS
# ============================================================

if "last_inspection_time" not in st.session_state:
    st.session_state["last_inspection_time"] = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="belt-hero">
        <div class="belt-brandline">
            <div class="belt-kicker">Condition Intelligence Platform</div>
        </div>
        <h1 style="color:#f8fafc !important; -webkit-text-fill-color:#f8fafc !important;">BeltGuard</h1>
        <div class="belt-title-rule"></div>
        <p>Conveyor condition intelligence for safer, more reliable operations.</p>
        <div class="belt-live"><span class="belt-dot"></span> System online</div>
    </div>
    """,
    unsafe_allow_html=True
)

last_run = st.session_state.get("last_inspection_time")
last_run_text = last_run.strftime("%d %b %Y · %H:%M") if last_run else "No inspection in this session"

st.markdown(
    f"""
    <div class="ops-strip">
        <div><span class="ops-label">SERVICE</span><span class="ops-value online">ONLINE</span></div>
        <div><span class="ops-label">MODE</span><span class="ops-value">IMAGE INSPECTION</span></div>
        <div><span class="ops-label">LAST INSPECTION</span><span class="ops-value">{last_run_text}</span></div>
        <div><span class="ops-label">THRESHOLD</span><span class="ops-value">0.24</span></div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">Inspection workspace</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">Start an inspection, review current belt condition, and record the recommended maintenance action.</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding:0.5rem 0 1.2rem;">
        <div style="font-size:1.28rem;font-weight:800;color:white;letter-spacing:-.02em;">BELTGUARD</div>
        <div style="font-size:0.68rem;color:#7890a7;margin-top:0.25rem;text-transform:uppercase;letter-spacing:.12em;">
            Conveyor condition intelligence
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### Inspection controls")

confidence = st.sidebar.slider(
    "Detection confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.24,
    step=0.01
)

st.sidebar.caption("Operating reference: **0.24**")

st.sidebar.markdown("---")


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Inspection image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Inspection image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG.",
    label_visibility="collapsed"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def health_status_emoji(status):
    mapping = {
        "HEALTHY": "",
        "WARNING": "",
        "DEGRADED": "",
        "CRITICAL": ""
    }
    return mapping.get(status, "")


def health_status_message(status):
    mapping = {
        "HEALTHY": "Belt condition appears normal.",
        "WARNING": "Minor condition concerns detected. Continue monitoring.",
        "DEGRADED": "Belt condition requires planned inspection.",
        "CRITICAL": "Immediate inspection is recommended."
    }
    return mapping.get(
        status,
        "Review the inspection results."
    )


def overall_maintenance_priority(
    health_status,
    detections
):
    """
    Derive an overall maintenance recommendation from
    the health status and detected severity.
    """

    has_high = any(
        d.get("severity") == "High"
        for d in detections
    )

    if health_status == "CRITICAL":
        return "IMMEDIATE INSPECTION"

    if has_high:
        return "IMMEDIATE INSPECTION"

    if health_status in ["DEGRADED", "WARNING"]:
        return "SCHEDULE INSPECTION"

    return "MONITOR"


# ============================================================
# PREDICTIVE MAINTENANCE HELPERS
# ============================================================

FAILURE_MODEL_PATH = os.path.join(
    "models",
    "failure_risk_xgboost.pkl"
)

FAILURE_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear"
]

FAILURE_DISPLAY_NAMES = [
    "Air Temperature",
    "Process Temperature",
    "Rotational Speed",
    "Torque",
    "Tool Wear"
]


@st.cache_resource
def load_failure_model():
    """Load the trained XGBoost predictive-maintenance model."""
    if not os.path.exists(FAILURE_MODEL_PATH):
        return None
    return joblib.load(FAILURE_MODEL_PATH)


def explain_failure_risk(
    model,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear
):
    """Return failure probability and local SHAP contributions."""

    values = np.array([[
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear
    ]], dtype=float)

    sensor_df = pd.DataFrame(
        values,
        columns=FAILURE_FEATURES
    )

    probability = float(
        model.predict_proba(sensor_df)[0, 1]
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sensor_df)

    if isinstance(shap_values, list):
        local_values = np.asarray(shap_values[-1])[0]
    else:
        local_values = np.asarray(shap_values)[0]

    explanation = pd.DataFrame({
        "Feature": FAILURE_DISPLAY_NAMES,
        "SHAP Value": local_values
    })

    explanation["Absolute Impact"] = (
        explanation["SHAP Value"].abs()
    )

    explanation = explanation.sort_values(
        "Absolute Impact",
        ascending=False
    ).reset_index(drop=True)

    return probability, explanation


def plot_local_shap(explanation):
    """Create a local SHAP contribution chart."""

    plot_df = explanation.sort_values(
        "SHAP Value",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(8, 4.2))

    ax.barh(
        plot_df["Feature"],
        plot_df["SHAP Value"]
    )

    ax.axvline(0, linewidth=1)

    ax.set_xlabel("Contribution to estimated risk")
    ax.set_title(
        "Operating risk drivers"
    )

    fig.tight_layout()

    return fig


# ============================================================
# PREDICTIVE RESULT STATE
# ============================================================

if "failure_risk_result" not in st.session_state:
    st.session_state["failure_risk_result"] = None


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Display original image
    # --------------------------------------------------------

    image = Image.open(uploaded_file)

    st.markdown('<div class="section-title">Input Image</div>', unsafe_allow_html=True)

    preview_col1, preview_col2 = st.columns([1.55, 1])

    with preview_col1:
        st.image(image, use_container_width=True)

    with preview_col2:
        st.markdown(
            f"""
            <div class="info-card">
                <div style="font-size:0.8rem;color:#64748b;font-weight:700;
                text-transform:uppercase;letter-spacing:0.05em;">Inspection File</div>
                <div style="font-size:1.05rem;font-weight:700;color:#0f172a;
                margin-top:0.35rem;word-break:break-word;">{uploaded_file.name}</div>
                <div style="margin-top:0.9rem;color:#64748b;font-size:0.88rem;">
                    Ready for AI inspection.<br><br>
                    The selected confidence threshold is
                    <b>{confidence:.2f}</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # Process button
    # --------------------------------------------------------

    # Keep analysis results in session state so clicking
    # the predictive-maintenance button does not erase them.
    upload_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("upload_key") != upload_key:
        st.session_state["upload_key"] = upload_key
        st.session_state.pop("belt_result", None)
        st.session_state.pop("belt_output_image_path", None)

    analyze_clicked = st.button(
        "Run Belt Inspection",
        type="primary",
        use_container_width=True
    )

    if analyze_clicked:

        with st.spinner(
            "Analyzing conveyor belt..."
        ):

            # ------------------------------------------------
            # Temporary directories
            # ------------------------------------------------

            temp_input = tempfile.mkdtemp()
            temp_output = tempfile.mkdtemp()

            input_path = os.path.join(
                temp_input,
                uploaded_file.name
            )

            with open(
                input_path,
                "wb"
            ) as f:
                f.write(
                    uploaded_file.getbuffer()
                )

            # ------------------------------------------------
            # Model path
            # ------------------------------------------------

            model_path = os.path.join(
                "runs",
                "train",
                "belt_damage_improved",
                "weights",
                "best.pt"
            )

            if not os.path.exists(
                model_path
            ):

                st.error(
                    "Improved model not found:\n"
                    f"{model_path}"
                )

                st.stop()

            # ------------------------------------------------
            # Run BeltGuard pipeline
            # ------------------------------------------------

            run_pipeline(
                image_dir=temp_input,
                output_dir=temp_output,
                model_path=model_path,
                conf_threshold=confidence,
                use_roi=False,
                tta=False
            )

            # ------------------------------------------------
            # Output paths
            # ------------------------------------------------

            base_name = os.path.splitext(
                uploaded_file.name
            )[0]

            output_image_path = os.path.join(
                temp_output,
                base_name + ".jpg"
            )

            output_json_path = os.path.join(
                temp_output,
                base_name + ".json"
            )

            # ------------------------------------------------
            # Check JSON
            # ------------------------------------------------

            if not os.path.exists(
                output_json_path
            ):

                st.error(
                    "Analysis result was not generated."
                )

                st.stop()

            with open(
                output_json_path,
                "r"
            ) as f:
                result = json.load(f)
            st.session_state["belt_result"] = result
            st.session_state["belt_output_image_path"] = output_image_path
            st.session_state["last_inspection_time"] = datetime.now()

    # Render the complete analysis result from session state.
    # This survives Streamlit reruns caused by the failure-risk button.
    if st.session_state.get("belt_result") is not None:

        result = st.session_state["belt_result"]
        output_image_path = st.session_state.get("belt_output_image_path")
        detections = result.get("detections", [])

        # RESULTS
        # =================================================

        st.markdown(
            '<div class="result-banner"><div><div class="eyebrow">Inspection complete</div><div class="value">Condition assessment is ready</div></div><span class="priority-chip">RESULTS READY</span></div>',
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # Extract health/anomaly information
        # -------------------------------------------------

        health = result.get(
            "health",
            {}
        )

        anomaly = result.get(
            "anomaly_detection",
            {}
        )

        health_score = float(
            health.get(
                "score",
                100
            )
        )

        health_status = health.get(
            "status",
            "HEALTHY"
        )

        anomaly_score = anomaly.get(
            "anomaly_score"
        )

        anomaly_status = anomaly.get(
            "status",
            "UNKNOWN"
        )

        # -------------------------------------------------
        # Detection summary
        # -------------------------------------------------

        total_defects = len(
            detections
        )

        scratch_count = sum(
            1
            for d in detections
            if d.get("class_name") == "scratch"
        )

        edge_damage_count = sum(
            1
            for d in detections
            if d.get("class_name") == "edge_damage"
        )

        high_count = sum(
            1
            for d in detections
            if d.get("severity") == "High"
        )

        medium_count = sum(
            1
            for d in detections
            if d.get("severity") == "Medium"
        )

        low_count = sum(
            1
            for d in detections
            if d.get("severity") == "Low"
        )

        overall_priority = overall_maintenance_priority(
            health_status,
            detections
        )

        # =================================================
        # HEALTH OVERVIEW
        # =================================================

        st.markdown(
            '<div class="section-title">Condition overview</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-subtitle">Current condition based on the latest inspection image.</div>',
            unsafe_allow_html=True
        )

        health_col1, health_col2 = st.columns(
            [1, 2]
        )

        with health_col1:

            st.metric(
                "Condition Score",
                f"{health_score:.0f} / 100"
            )

            st.markdown(
                f"### "
                f"{health_status_emoji(health_status)} "
                f"{health_status}"
            )

        with health_col2:

            st.markdown(
                f"""
                **Condition Assessment**

                {health_status_message(health_status)}

                **Maintenance Recommendation:**  
                ### {overall_priority}
                """
            )

        # Health progress bar
        st.progress(
            max(
                0.0,
                min(
                    health_score / 100.0,
                    1.0
                )
            )
        )

        # =================================================
        # HEALTH SCORE COMPONENTS
        # =================================================

        st.markdown(
            "### Condition indicators"
        )

        h1, h2, h3 = st.columns(3)

        with h1:
            st.metric(
                "Defect condition",
                f"{health.get('defect_condition', 100):.0f}/100"
            )

        with h2:
            st.metric(
                "Severity condition",
                f"{health.get('severity_condition', 100):.0f}/100"
            )

        with h3:
            st.metric(
                "Visual anomaly",
                f"{health.get('anomaly_condition', 100):.0f}/100"
            )

        st.caption(
            "Visual Condition Score = "
            "25% Defect Condition + "
            "45% Severity Condition + "
            "30% Anomaly Condition. "
            "Each component is normalized independently."
        )

        # =================================================
        # ANOMALY MONITORING
        # =================================================

        st.markdown('<div class="section-title">Visual condition signals</div>', unsafe_allow_html=True)

        a1, a2, a3 = st.columns(3)

        with a1:
            if anomaly_status == "ANOMALOUS":
                st.error(
                    "ANOMALOUS"
                )
            elif anomaly_status == "NORMAL":
                st.success(
                    "NORMAL"
                )
            else:
                st.warning(
                    f"Status: {anomaly_status}"
                )

        with a2:
            if anomaly_score is not None:
                st.metric(
                    "Visual deviation",
                    f"{float(anomaly_score):.6f}"
                )
            else:
                st.metric(
                    "Visual deviation",
                    "N/A"
                )

        with a3:
            threshold = anomaly.get(
                "threshold",
                0
            )

            st.metric(
                "Reference level",
                f"{float(threshold):.6f}"
            )

        st.caption(
            "Higher reconstruction error indicates that "
            "the belt image is visually more unusual than "
            "patterns learned by the autoencoder."
        )

        # =================================================
        # PREDICTIVE MAINTENANCE / FAILURE RISK
        # =================================================

        st.markdown('<div class="section-title">Operating condition</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Enter current operating readings when equipment telemetry is available.</div>',
            unsafe_allow_html=True
        )

        st.info(
            "This XGBoost model is a separate sensor-based "
            "predictive-maintenance benchmark. Enter machine "
            "operating values below to estimate failure risk. "
            "These values are not extracted from the uploaded "
            "conveyor image."
        )

        failure_model = load_failure_model()

        if failure_model is None:

            st.warning(
                "Failure-risk model not found. Expected: "
                "models/failure_risk_xgboost.pkl"
            )

        else:

            st.markdown("**Operating readings**")

            pm1, pm2, pm3 = st.columns(3)

            with pm1:

                air_temperature = st.number_input(
                    "Air Temperature [K]",
                    min_value=250.0,
                    max_value=350.0,
                    value=298.1,
                    step=0.1
                )

                process_temperature = st.number_input(
                    "Process Temperature [K]",
                    min_value=250.0,
                    max_value=360.0,
                    value=308.6,
                    step=0.1
                )

            with pm2:

                rotational_speed = st.number_input(
                    "Rotational Speed [rpm]",
                    min_value=500,
                    max_value=3000,
                    value=1500,
                    step=10
                )

                torque = st.number_input(
                    "Torque [Nm]",
                    min_value=0.0,
                    max_value=100.0,
                    value=40.0,
                    step=0.5
                )

            with pm3:

                tool_wear = st.number_input(
                    "Tool Wear [min]",
                    min_value=0,
                    max_value=300,
                    value=100,
                    step=5
                )

                st.caption(
                    "Use actual equipment sensor readings "
                    "when available."
                )

            if st.button(
                "Assess operating risk",
                key="failure_risk_button",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Assessing operating condition..."
                ):

                    failure_probability, shap_explanation = (
                        explain_failure_risk(
                            failure_model,
                            air_temperature,
                            process_temperature,
                            rotational_speed,
                            torque,
                            tool_wear
                        )
                    )

                risk_percent = failure_probability * 100

                r1, r2 = st.columns(2)

                with r1:
                    st.metric(
                        "Failure risk",
                        f"{risk_percent:.2f}%"
                    )

                with r2:

                    if risk_percent >= 60:
                        risk_label = "HIGH"
                    elif risk_percent >= 30:
                        risk_label = "MODERATE"
                    else:
                        risk_label = "LOW"

                    st.metric(
                        "Risk level",
                        risk_label
                    )

                st.session_state["failure_risk_result"] = {
                    "risk_percent": risk_percent,
                    "risk_label": risk_label,
                    "shap_explanation": shap_explanation
                }

                st.caption(
                    "Risk levels are screening indicators, not calibrated maintenance limits."
                )

                st.markdown(
                    "#### Risk drivers"
                )

                positive = shap_explanation[
                    shap_explanation["SHAP Value"] > 0
                ]

                negative = shap_explanation[
                    shap_explanation["SHAP Value"] < 0
                ]

                if not positive.empty:

                    top_positive = positive.iloc[0]

                    st.write(
                        f"**{top_positive['Feature']}** "
                        f"contributed toward higher predicted "
                        f"failure risk."
                    )

                if not negative.empty:

                    top_negative = negative.iloc[0]

                    st.write(
                        f"**{top_negative['Feature']}** "
                        f"contributed toward lower predicted "
                        f"failure risk."
                    )

                st.pyplot(
                    plot_local_shap(shap_explanation),
                    use_container_width=True
                )

                st.markdown(
                    "#### Risk drivers"
                )

                display_explanation = shap_explanation[
                    ["Feature", "SHAP Value"]
                ].copy()

                display_explanation["Effect"] = (
                    display_explanation["SHAP Value"].apply(
                        lambda value:
                        "Higher risk"
                        if value > 0
                        else "Lower risk"
                        if value < 0
                        else "Neutral"
                    )
                )

                display_explanation["SHAP Value"] = (
                    display_explanation["SHAP Value"].round(4)
                )

                st.dataframe(
                    display_explanation,
                    hide_index=True,
                    use_container_width=True
                )

                with st.expander(
                    "Technical details"
                ):

                    st.write(
                        """
                        **Model:** XGBoost binary classifier

                        **Purpose:** Estimate machine-failure
                        probability from operating/sensor variables.

                        **Features:** Air Temperature,
                        Process Temperature, Rotational Speed,
                        Torque, and Tool Wear.

                        **Explainability:** Local SHAP values
                        show how each input contributes to the
                        individual prediction.

                        **Data source:** UCI AI4I 2020
                        predictive-maintenance benchmark dataset.

                        **Important:** The AI4I dataset is separate
                        from the conveyor-belt image dataset.
                        The failure-risk model should therefore not
                        be described as predicting failure directly
                        from the uploaded image.
                        """
                    )


        # =================================================
        # =================================================
        # Use the anomaly result already produced by the analysis pipeline.
        anomaly_result = (
            anomaly if isinstance(anomaly, dict) else {}
        )

        # MAINTENANCE DECISION SUPPORT
        # =================================================

        st.markdown("---")
        st.markdown('<div class="section-title">Maintenance action</div>', unsafe_allow_html=True)

        st.caption(
            "Decision-support summary combining visual inspection findings "
            "with the separate sensor-based predictive-maintenance benchmark."
        )

        decision_col1, decision_col2 = st.columns(2)

        with decision_col1:
            st.markdown("### Inspection condition")

            st.metric(
                "Visual Condition",
                health_status
            )

            st.metric(
                "Detected Defects",
                len(detections)
            )

            high_severity_count = sum(
                1
                for det in detections
                if det.get("severity", "").lower() == "high"
            )

            st.metric(
                "High-Severity Defects",
                high_severity_count
            )

            anomaly_status = anomaly_result.get(
                "status",
                "UNKNOWN"
            )

            st.metric(
                "Visual Anomaly",
                anomaly_status
            )

        with decision_col2:
            st.markdown("### Operating condition")

            failure_state = st.session_state.get("failure_risk_result")

            if failure_state is not None:
                st.metric(
                    "Failure risk",
                    f"{failure_state['risk_percent']:.2f}%"
                )

                st.metric(
                    "Risk level",
                    failure_state["risk_label"]
                )
            else:
                st.info(
                    "Calculate the sensor-based failure risk above "
                    "to include it in this summary."
                )

        st.markdown("### Recommended Action")

        if (
            health_status == "CRITICAL"
            or high_severity_count >= 3
            or anomaly_status == "ANOMALOUS"
        ):
            st.error("IMMEDIATE VISUAL INSPECTION")

            st.write(
                "The visual inspection indicates significant belt condition "
                "concerns. Inspect the affected belt regions before continued "
                "operation."
            )

        elif health_status == "DEGRADED":
            st.warning("SCHEDULE INSPECTION")

            st.write(
                "The visual inspection indicates degraded belt condition. "
                "Schedule a detailed inspection and continue monitoring."
            )

        else:
            st.success("MONITOR")

            st.write(
                "No immediate visual maintenance action is indicated. "
                "Continue routine monitoring."
            )

        st.caption(
            "This recommendation is based on visual inspection indicators "
            "and is not a calibrated probability of equipment failure or "
            "remaining useful life (RUL)."
        )

        # INSPECTION SUMMARY
        # =================================================

        st.markdown('<div class="section-title">Inspection snapshot</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Defects",
                total_defects
            )

        with col2:
            st.metric(
                "Scratch",
                scratch_count
            )

        with col3:
            st.metric(
                "Edge Damage",
                edge_damage_count
            )

        with col4:
            st.metric(
                "High Severity",
                high_count
            )

        # =================================================
        # OUTPUT IMAGE
        # =================================================

        st.markdown('<div class="section-title">Inspection result</div>', unsafe_allow_html=True)

        if os.path.exists(
            output_image_path
        ):

            result_image = Image.open(
                output_image_path
            )

            st.image(
                result_image,
                use_container_width=True
            )

        # =================================================
        # SEVERITY SUMMARY
        # =================================================

        st.markdown('<div class="section-title">Severity profile</div>', unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)

        with s1:
            st.metric(
                "High",
                high_count
            )

        with s2:
            st.metric(
                "Medium",
                medium_count
            )

        with s3:
            st.metric(
                "Low",
                low_count
            )

        # =================================================
        # DETECTION TABLE
        # =================================================

        st.markdown('<div class="section-title">Findings</div>', unsafe_allow_html=True)

        if detections:

            for index, detection in enumerate(
                detections,
                start=1
            ):

                class_name = detection.get(
                    "class_name",
                    "Unknown"
                )

                confidence_value = detection.get(
                    "confidence",
                    0
                )

                severity = detection.get(
                    "severity",
                    "Unknown"
                )

                severity_score = detection.get(
                    "severity_score",
                    0
                )

                maintenance = detection.get(
                    "maintenance_priority",
                    "MONITOR"
                )

                with st.container():

                    c1, c2, c3, c4 = st.columns(
                        [2, 1, 1, 3]
                    )

                    with c1:
                        st.write(
                            f"**{index}. "
                            f"{class_name.upper()}**"
                        )

                    with c2:
                        st.write(
                            f"Confidence: "
                            f"**{float(confidence_value):.0%}**"
                        )

                    with c3:
                        st.write(
                            f"Severity: "
                            f"**{severity}**"
                        )

                    with c4:
                        st.write(
                            f"Action: "
                            f"**{maintenance}**"
                        )

                    st.progress(
                        min(
                            float(severity_score) / 100,
                            1.0
                        )
                    )

                    st.divider()

        else:

            st.success(
                "No belt damage detected above "
                "the selected confidence threshold."
            )

        # =================================================
        # METHODOLOGY
        # =================================================

        with st.expander(
            "Model & Methodology"
        ):

            st.write(
                """
                **Inspection engine:** Object detection

                **Detected Classes:**
                - Scratch
                - Edge Damage

                **Anomaly Detection:** ROI convolutional
                autoencoder using reconstruction error.

                **Severity:** Visual severity proxy based
                on detected bounding-box characteristics.

                **Condition Score:** Normalized weighted
                visual condition score based on defect
                burden, visual severity, and visual anomaly
                reconstruction error.

                **Weights:** 25% defect condition,
                45% severity condition, and 30% anomaly
                condition. Each component is normalized
                independently before combining them.

                **Maintenance Priority:** Combines health
                condition and detected severity.

                **Important:** The Health Score is not a
                failure probability, Remaining Useful Life
                (RUL), or learned physical equipment-health
                prediction.

                **Important:** The current health score is
                an operational scoring layer, not a learned
                physical equipment-health prediction. The
                current dataset does not contain human-labeled
                severity, failure-history, or remaining-useful-
                life labels.
                """
            )


        # ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer-line">
        BELTGUARD &nbsp;·&nbsp; Conveyor condition intelligence &nbsp;·&nbsp; Inspection console
    </div>
    """,
    unsafe_allow_html=True
)
