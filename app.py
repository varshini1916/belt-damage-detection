import os
import json
import tempfile

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
    page_icon="🔧",
    layout="wide"
)


# ============================================================
# PROFESSIONAL UI THEME
# ============================================================

st.markdown(
    """
    <style>
    /* Main application */
    .stApp {
        background: #f5f7fb;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #243044;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    section[data-testid="stSidebar"] .stSlider > div > div > div {
        background: #60a5fa;
    }


    /* Main content typography - explicit contrast for light dashboard */
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] strong,
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] em,
    section[data-testid="stMain"] label,
    section[data-testid="stMain"] label p,
    section[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stMain"] [data-testid="stWidgetLabel"] label {
        color: #1e293b !important;
    }

    section[data-testid="stMain"] h1,
    section[data-testid="stMain"] h2,
    section[data-testid="stMain"] h3,
    section[data-testid="stMain"] h4,
    section[data-testid="stMain"] h5,
    section[data-testid="stMain"] h6 {
        color: #0f172a !important;
    }

    section[data-testid="stMain"] [data-testid="stCaptionContainer"],
    section[data-testid="stMain"] [data-testid="stCaptionContainer"] p {
        color: #64748b !important;
    }

    /* Form controls - high contrast and readable */
    section[data-testid="stMain"] input,
    section[data-testid="stMain"] textarea,
    section[data-testid="stMain"] [data-baseweb="input"] input,
    section[data-testid="stMain"] [data-baseweb="base-input"] input {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        background-color: #ffffff !important;
        caret-color: #2563eb !important;
        opacity: 1 !important;
    }

    section[data-testid="stMain"] [data-baseweb="input"],
    section[data-testid="stMain"] [data-baseweb="base-input"],
    section[data-testid="stMain"] [data-testid="stNumberInput"] [data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    section[data-testid="stMain"] [data-testid="stNumberInput"] button {
        color: #334155 !important;
        background: #f8fafc !important;
        border-color: #e2e8f0 !important;
    }

    section[data-testid="stMain"] [data-testid="stNumberInput"] button:hover {
        background: #e2e8f0 !important;
    }

    section[data-testid="stMain"] [data-testid="stNumberInput"] input::placeholder {
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important;
    }

    /* Keep info/success/warning/error boxes readable */
    section[data-testid="stMain"] [data-testid="stAlert"] p,
    section[data-testid="stMain"] [data-testid="stAlert"] div {
        color: inherit;
    }

    /* Sidebar keeps its intentional dark theme */
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #e5e7eb !important;
    }

    /* Hero */
    .belt-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #2563eb 100%);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.14);
        color: white;
    }

    .belt-hero h1 {
        margin: 0;
        font-size: 2.35rem;
        font-weight: 750;
        letter-spacing: -0.03em;
        color: white;
    }

    .belt-hero p {
        margin: 0.55rem 0 0;
        color: #dbeafe;
        font-size: 1.03rem;
        line-height: 1.6;
    }

    .belt-badge {
        display: inline-block;
        padding: 0.32rem 0.72rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        color: #dbeafe;
        font-size: 0.78rem;
        font-weight: 650;
        margin-bottom: 0.8rem;
    }

    /* Section headings */
    .section-title {
        font-size: 1.35rem;
        font-weight: 720;
        color: #0f172a;
        margin: 1.5rem 0 0.75rem;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 0.92rem;
        margin-top: -0.45rem;
        margin-bottom: 1rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.95rem 1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 750;
    }

    /* Containers / cards */
    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.045);
        margin-bottom: 1rem;
    }

    .status-card {
        background: white;
        border-radius: 16px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
    }

    .status-label {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .status-value {
        color: #0f172a;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: white;
        border: 1.5px dashed #94a3b8;
        border-radius: 16px;
        padding: 0.5rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.65rem;
        border: 1px solid #cbd5e1;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 14px rgba(15, 23, 42, 0.12);
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: #2563eb;
        border-color: #2563eb;
        color: white;
    }

    .stButton > button[kind="primary"] {
        background: #2563eb !important;
        color: #ffffff !important;
        border-color: #2563eb !important;
    }

    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span {
        color: #ffffff !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Dividers */
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 1.5rem 0;
    }

    /* Footer */
    .belt-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.82rem;
        padding: 1.5rem 0 0.5rem;
    }

    /* Hide Streamlit default decoration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="belt-hero">
        <div class="belt-badge">AI-POWERED INDUSTRIAL INSPECTION</div>
        <h1>🔧 BeltGuard</h1>
        <p>
            Intelligent conveyor-belt monitoring using visual damage detection,
            anomaly analysis, severity estimation and maintenance decision support.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SYSTEM STATUS
# ============================================================

status_cols = st.columns(3)

with status_cols[0]:
    st.markdown(
        '''
        <div class="status-card">
            <div class="status-label">Inspection Engine</div>
            <div class="status-value" style="font-size:1.25rem;">🟢 READY</div>
            <div style="color:#64748b;font-size:0.82rem;">YOLOv8 defect detection</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

with status_cols[1]:
    st.markdown(
        '''
        <div class="status-card">
            <div class="status-label">Anomaly Detector</div>
            <div class="status-value" style="font-size:1.25rem;">🟢 READY</div>
            <div style="color:#64748b;font-size:0.82rem;">ROI autoencoder analysis</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

with status_cols[2]:
    st.markdown(
        '''
        <div class="status-card">
            <div class="status-label">Risk Model</div>
            <div class="status-value" style="font-size:1.25rem;">🟢 READY</div>
            <div style="color:#64748b;font-size:0.82rem;">XGBoost + SHAP benchmark</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">📤 Inspection Console</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">Upload a conveyor-belt inspection image and run the AI analysis pipeline.</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding:0.5rem 0 1.2rem;">
        <div style="font-size:1.35rem;font-weight:800;color:white;">🔧 BeltGuard</div>
        <div style="font-size:0.78rem;color:#94a3b8;margin-top:0.2rem;">
            Conveyor Intelligence Platform
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### ⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.24,
    step=0.01
)

st.sidebar.caption("Recommended operating threshold: **0.24**")

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="margin-top:1rem;padding:1rem;border:1px solid #334155;
    border-radius:12px;background:#172033;">
        <div style="font-weight:700;margin-bottom:0.65rem;">System Stack</div>
        <div style="font-size:0.82rem;line-height:1.75;color:#cbd5e1;">
            <b>Detection</b> · YOLOv8<br>
            <b>Classes</b> · Scratch, Edge Damage<br>
            <b>Anomaly</b> · ROI Autoencoder<br>
            <b>Severity</b> · Visual Proxy<br>
            <b>Health</b> · Visual Condition Score<br>
            <b>Prediction</b> · XGBoost + SHAP
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload conveyor-belt image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def health_status_emoji(status):
    mapping = {
        "HEALTHY": "🟢",
        "WARNING": "🟡",
        "DEGRADED": "🟠",
        "CRITICAL": "🔴"
    }
    return mapping.get(status, "⚪")


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

    ax.set_xlabel("SHAP contribution")
    ax.set_title(
        "Local SHAP Explanation for Failure Risk"
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

    st.markdown('<div class="section-title">📷 Input Image</div>', unsafe_allow_html=True)

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
        "🔍  Run Belt Inspection",
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

    # Render the complete analysis result from session state.
    # This survives Streamlit reruns caused by the failure-risk button.
    if st.session_state.get("belt_result") is not None:

        result = st.session_state["belt_result"]
        output_image_path = st.session_state.get("belt_output_image_path")
        detections = result.get("detections", [])

        # RESULTS
        # =================================================

        st.success("Inspection completed successfully.")

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
            '<div class="section-title">🩺 Belt Health Overview</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-subtitle">Visual condition assessment generated from detected defects, severity and ROI anomaly evidence.</div>',
            unsafe_allow_html=True
        )

        health_col1, health_col2 = st.columns(
            [1, 2]
        )

        with health_col1:

            st.metric(
                "Health Score",
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
                ### 🛠️ {overall_priority}
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
            "### 📐 Health Score Components"
        )

        h1, h2, h3 = st.columns(3)

        with h1:
            st.metric(
                "Defect Condition",
                f"{health.get('defect_condition', 100):.0f}/100"
            )

        with h2:
            st.metric(
                "Severity Condition",
                f"{health.get('severity_condition', 100):.0f}/100"
            )

        with h3:
            st.metric(
                "Anomaly Condition",
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

        st.markdown('<div class="section-title">🧠 ROI Anomaly Detection</div>', unsafe_allow_html=True)

        a1, a2, a3 = st.columns(3)

        with a1:
            if anomaly_status == "ANOMALOUS":
                st.error(
                    "⚠️ ANOMALOUS"
                )
            elif anomaly_status == "NORMAL":
                st.success(
                    "✅ NORMAL"
                )
            else:
                st.warning(
                    f"Status: {anomaly_status}"
                )

        with a2:
            if anomaly_score is not None:
                st.metric(
                    "Reconstruction Error",
                    f"{float(anomaly_score):.6f}"
                )
            else:
                st.metric(
                    "Reconstruction Error",
                    "N/A"
                )

        with a3:
            threshold = anomaly.get(
                "threshold",
                0
            )

            st.metric(
                "Anomaly Threshold",
                f"{float(threshold):.6f}"
            )

        st.caption(
            "Higher reconstruction error indicates that "
            "the belt ROI is visually more unusual than "
            "patterns learned by the autoencoder."
        )

        # =================================================
        # PREDICTIVE MAINTENANCE / FAILURE RISK
        # =================================================

        st.markdown('<div class="section-title">🔮 Predictive Maintenance Analytics</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Complementary sensor-based failure-risk benchmark with local model explainability.</div>',
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

            st.markdown("**Sensor / Operating Inputs**")

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
                "📈  Calculate Failure Risk",
                key="failure_risk_button",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Calculating failure risk and SHAP explanation..."
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
                        "Predicted Failure Risk",
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
                        "Model Risk Band",
                        risk_label
                    )

                st.session_state["failure_risk_result"] = {
                    "risk_percent": risk_percent,
                    "risk_label": risk_label,
                    "shap_explanation": shap_explanation
                }

                st.caption(
                    "Risk-band thresholds are presentation "
                    "thresholds only; they are not calibrated "
                    "maintenance decision limits."
                )

                st.markdown(
                    "#### 🧩 Why did the model make this prediction?"
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
                        f"🔴 **{top_positive['Feature']}** "
                        f"contributed toward higher predicted "
                        f"failure risk."
                    )

                if not negative.empty:

                    top_negative = negative.iloc[0]

                    st.write(
                        f"🟢 **{top_negative['Feature']}** "
                        f"contributed toward lower predicted "
                        f"failure risk."
                    )

                st.pyplot(
                    plot_local_shap(shap_explanation),
                    use_container_width=True
                )

                st.markdown(
                    "#### 📊 Feature Contributions"
                )

                display_explanation = shap_explanation[
                    ["Feature", "SHAP Value"]
                ].copy()

                display_explanation["Direction"] = (
                    display_explanation["SHAP Value"].apply(
                        lambda value:
                        "Increases risk"
                        if value > 0
                        else "Decreases risk"
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
                    "ℹ️ Predictive Maintenance Model Details"
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
        st.markdown('<div class="section-title">🛠️ Maintenance Decision Support</div>', unsafe_allow_html=True)

        st.caption(
            "Decision-support summary combining visual inspection findings "
            "with the separate sensor-based predictive-maintenance benchmark."
        )

        decision_col1, decision_col2 = st.columns(2)

        with decision_col1:
            st.markdown("### 📷 Visual Inspection")

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
            st.markdown("### 🔮 Sensor-Based Prediction")

            failure_state = st.session_state.get("failure_risk_result")

            if failure_state is not None:
                st.metric(
                    "Sensor Failure Risk",
                    f"{failure_state['risk_percent']:.2f}%"
                )

                st.metric(
                    "Model Risk Band",
                    failure_state["risk_label"]
                )
            else:
                st.info(
                    "Calculate the sensor-based failure risk above "
                    "to include it in this summary."
                )

        st.markdown("### 🛠️ Recommended Action")

        if (
            health_status == "CRITICAL"
            or high_severity_count >= 3
            or anomaly_status == "ANOMALOUS"
        ):
            st.error("🔴 IMMEDIATE VISUAL INSPECTION")

            st.write(
                "The visual inspection indicates significant belt condition "
                "concerns. Inspect the affected belt regions before continued "
                "operation."
            )

        elif health_status == "DEGRADED":
            st.warning("🟠 SCHEDULE INSPECTION")

            st.write(
                "The visual inspection indicates degraded belt condition. "
                "Schedule a detailed inspection and continue monitoring."
            )

        else:
            st.success("🟢 MONITOR")

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

        st.markdown('<div class="section-title">📊 Inspection Summary</div>', unsafe_allow_html=True)

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

        st.markdown('<div class="section-title">🔎 Detection Result</div>', unsafe_allow_html=True)

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

        st.markdown('<div class="section-title">⚠️ Severity Distribution</div>', unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)

        with s1:
            st.metric(
                "🔴 High",
                high_count
            )

        with s2:
            st.metric(
                "🟠 Medium",
                medium_count
            )

        with s3:
            st.metric(
                "🟢 Low",
                low_count
            )

        # =================================================
        # DETECTION TABLE
        # =================================================

        st.markdown('<div class="section-title">📋 Detailed Detection Results</div>', unsafe_allow_html=True)

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
            "ℹ️ Model & Methodology"
        ):

            st.write(
                """
                **Detection Model:** YOLOv8

                **Detected Classes:**
                - Scratch
                - Edge Damage

                **Anomaly Detection:** ROI convolutional
                autoencoder using reconstruction error.

                **Severity:** Visual severity proxy based
                on detected bounding-box characteristics.

                **Health Score:** Normalized weighted
                visual condition score based on defect
                burden, visual severity, and ROI anomaly
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
        # HOW BELTGUARD WORKS
        # ============================================================

        st.markdown("---")
        st.markdown(
            '<div class="section-title">🧭 How BeltGuard Works</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-subtitle">A layered AI workflow for conveyor-belt inspection and maintenance decision support.</div>',
            unsafe_allow_html=True
        )

        with st.expander("View the BeltGuard inspection workflow", expanded=False):
            wf1, wf2, wf3 = st.columns(3)

            with wf1:
                st.markdown(
                    '''
                    <div class="info-card">
                        <b>1. Visual Inspection</b><br>
                        <span style="color:#64748b;">
                        Upload a conveyor image and detect visible belt defects using the trained YOLOv8 model.
                        </span>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '''
                    <div class="info-card">
                        <b>2. Defect Classification</b><br>
                        <span style="color:#64748b;">
                        Identify detected regions as scratch or edge damage and estimate a visual severity level.
                        </span>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            with wf2:
                st.markdown(
                    '''
                    <div class="info-card">
                        <b>3. Visual Anomaly Analysis</b><br>
                        <span style="color:#64748b;">
                        The ROI convolutional autoencoder compares belt appearance against learned visual patterns using reconstruction error.
                        </span>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '''
                    <div class="info-card">
                        <b>4. Condition Scoring</b><br>
                        <span style="color:#64748b;">
                        Defect burden, visual severity and anomaly evidence are combined into a normalized visual condition score.
                        </span>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            with wf3:
                st.markdown(
                    '''
                    <div class="info-card">
                        <b>5. Maintenance Decision Support</b><br>
                        <span style="color:#64748b;">
                        Inspection findings are translated into monitor, schedule-inspection or immediate-inspection guidance.
                        </span>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '''
                    <div class="info-card">
                        <b>6. Sensor-Risk Benchmark</b><br>
                        <span style="color:#64748b;">
                        Optional operating values are evaluated by a separate XGBoost model and explained with SHAP.
                        </span>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            st.info(
                "Deployment note: the current application performs image-based inspection on uploaded images. "
                "The XGBoost failure-risk component is a separate sensor-based benchmark and does not derive sensor values from the image. "
                "This application should therefore be described as AI-assisted inspection and maintenance decision support, "
                "not as a live equipment sensor system or RUL predictor."
            )

        # ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="belt-footer">
        <b>BeltGuard</b> · AI-Based Conveyor Belt Inspection &nbsp;|&nbsp;
        YOLOv8 · ROI Autoencoder · XGBoost · SHAP
        <br>
        <span style="font-size:0.75rem;">AI-assisted inspection and maintenance decision support</span>
    </div>
    """,
    unsafe_allow_html=True
)
