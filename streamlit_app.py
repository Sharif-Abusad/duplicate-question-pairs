# import streamlit as st
# from app.pipeline import create_feature_vector
# from app.inference import predict_duplicate

# st.header('Duplicate Question Pairs')

# q1 = st.text_input('Enter Question 1')
# q2 = st.text_input('Enter Question 2')

# if st.button('Predict'):
#     query = create_feature_vector(q1, q2)
#     result = predict_duplicate(query)

#     if result["duplicate"]:
#         st.success("Duplicate Questions")
#     else:
#         st.error("Not Duplicate Questions")

#     st.write(f"Probability: {result['probability']}")
#     st.write(f"Confidence: {result['confidence']}")
# import streamlit as st

# from app.pipeline import create_feature_vector
# from app.inference import predict_duplicate
# st.markdown("""
# <style>
# .block-container{
#     padding-top:2rem;
#     padding-bottom:2rem;
# }

# h1 {
#     text-align:center;
# }

# [data-testid="stMetric"] {
#     border:1px solid #e6e6e6;
#     border-radius:10px;
#     padding:15px;
# }
# </style>
# """, unsafe_allow_html=True)
# # -------------------------------------------------------
# # Page Configuration
# # -------------------------------------------------------
# st.set_page_config(
#     page_title="Duplicate Question Pair Detector",
#     page_icon="🔍",
#     layout="wide"
# )

# # -------------------------------------------------------
# # Header
# # -------------------------------------------------------
# st.title("🔍 Duplicate Question Pair Detector")
# st.markdown(
#     """
#     Detect whether two questions express the same intent using
#     Machine Learning and advanced NLP feature engineering.
#     """
# )

# st.divider()

# # -------------------------------------------------------
# # Input Section
# # -------------------------------------------------------
# col1, col2 = st.columns(2)

# with col1:
#     q1 = st.text_area(
#         "Question 1",
#         placeholder="Enter the first question...",
#         height=120
#     )

# with col2:
#     q2 = st.text_area(
#         "Question 2",
#         placeholder="Enter the second question...",
#         height=120
#     )

# st.markdown("")

# # -------------------------------------------------------
# # Prediction Button
# # -------------------------------------------------------
# if st.button("🚀 Analyze Questions", use_container_width=True):

#     if not q1.strip() or not q2.strip():
#         st.warning("Please enter both questions.")
#         st.stop()

#     with st.spinner("Analyzing question pair..."):

#         query = create_feature_vector(q1, q2)
#         result = predict_duplicate(query)

#     probability = result["probability"]
#     confidence = result["confidence"]

#     st.divider()

#     # ---------------------------------------------------
#     # Result Banner
#     # ---------------------------------------------------
#     if result["duplicate"]:
#         st.success("✅ Duplicate Question Pair")
#     else:
#         st.error("❌ Not a Duplicate Question Pair")

#     st.markdown("")

#     # ---------------------------------------------------
#     # Metrics
#     # ---------------------------------------------------
#     col1, col2 = st.columns(2)

#     with col1:
#         st.metric(
#             label="Duplicate Probability",
#             value=f"{probability:.2%}"
#         )

#     with col2:
#         st.metric(
#             label="Confidence Level",
#             value=confidence.capitalize()
#         )

#     st.markdown("### Prediction Confidence")

#     st.progress(float(probability))

#     # ---------------------------------------------------
#     # Additional Details
#     # ---------------------------------------------------
#     with st.expander("View Prediction Details"):
#         st.json(result)

# # -------------------------------------------------------
# # Footer
# # -------------------------------------------------------
# st.divider()

# st.caption(
#     "Built with Streamlit • Random Forest • NLP Feature Engineering"
# )
import streamlit as st

# -------------------------------------------------------
# Page Configuration — must be first Streamlit call
# -------------------------------------------------------
st.set_page_config(
    page_title="Duplicate Question Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------
# Global Styles
# -------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ─────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0D1117;
    color: #E6EDF3;
}

.block-container {
    padding: 2.5rem 3rem 3rem 3rem;
    max-width: 1100px;
}

/* ── Hide Streamlit chrome ────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Typography ───────────────────────────────────── */
h1, h2, h3 { color: #E6EDF3; }

/* ── Divider ──────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid #21262D;
    margin: 1.75rem 0;
}

/* ── Hero Header ──────────────────────────────────── */
.hero {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.25rem;
}

.hero-icon {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}

.hero-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #E6EDF3;
    margin: 0;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 0.9rem;
    color: #8B949E;
    margin: 0.35rem 0 0 0;
    font-weight: 400;
    line-height: 1.5;
}

/* ── Input Cards ──────────────────────────────────── */
.input-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8B949E;
    margin-bottom: 0.4rem;
    display: block;
}

.input-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 6px;
}

.badge-q1 { background: #1C2A3A; color: #58A6FF; border: 1px solid #1F3A5C; }
.badge-q2 { background: #1E1C2E; color: #A78BFA; border: 1px solid #2D2050; }

textarea {
    background-color: #161B22 !important;
    border: 1px solid #30363D !important;
    border-radius: 10px !important;
    color: #E6EDF3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease !important;
    resize: none !important;
}

textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    outline: none !important;
}

textarea::placeholder { color: #484F58 !important; }

/* ── Primary Button ───────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.65rem 1.5rem !important;
    transition: opacity 0.2s ease, transform 0.1s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}

.stButton > button:active { transform: translateY(0) !important; }

/* ── Result Banner ────────────────────────────────── */
.result-banner {
    border-radius: 12px;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
}

.result-duplicate {
    background: rgba(35, 134, 54, 0.15);
    border: 1px solid rgba(35, 134, 54, 0.4);
    color: #3FB950;
}

.result-not-duplicate {
    background: rgba(218, 54, 51, 0.12);
    border: 1px solid rgba(218, 54, 51, 0.35);
    color: #F85149;
}

.result-icon { font-size: 1.4rem; }

/* ── Metric Cards ─────────────────────────────────── */
.metric-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    height: 100%;
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8B949E;
    margin-bottom: 0.6rem;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: #E6EDF3;
    line-height: 1;
}

.metric-badge {
    display: inline-block;
    margin-top: 0.6rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

.badge-high   { background: rgba(35,134,54,0.15); color: #3FB950; border: 1px solid rgba(35,134,54,0.3); }
.badge-medium { background: rgba(210,153,34,0.15); color: #D29922; border: 1px solid rgba(210,153,34,0.3); }
.badge-low    { background: rgba(218,54,51,0.12);  color: #F85149; border: 1px solid rgba(218,54,51,0.3); }

/* ── Progress Bar ─────────────────────────────────── */
.prob-bar-wrap {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
}

.prob-bar-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.75rem;
}

.prob-bar-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8B949E;
}

.prob-bar-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #E6EDF3;
}

.prob-track {
    background: #21262D;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}

.prob-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}

/* ── Expander ─────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 10px !important;
    color: #8B949E !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

.streamlit-expanderContent {
    background: #0D1117 !important;
    border: 1px solid #21262D !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Warning / Info ───────────────────────────────── */
.stAlert {
    background: #161B22 !important;
    border-radius: 10px !important;
    border: 1px solid #30363D !important;
    color: #E6EDF3 !important;
}

/* ── Footer ───────────────────────────────────────── */
.footer {
    margin-top: 2.5rem;
    padding-top: 1.25rem;
    border-top: 1px solid #21262D;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.footer-left {
    font-size: 0.78rem;
    color: #484F58;
}

.footer-pills {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.pill {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.7rem;
    font-weight: 500;
    color: #6E7681;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Spinner ──────────────────────────────────────── */
.stSpinner > div { border-top-color: #6366F1 !important; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# Lazy imports (after page config)
# -------------------------------------------------------
from app.pipeline import create_feature_vector   # noqa: E402
from app.inference import predict_duplicate       # noqa: E402


# -------------------------------------------------------
# Helper: confidence badge CSS class
# -------------------------------------------------------
def _confidence_class(level: str) -> str:
    return {"high": "badge-high", "medium": "badge-medium"}.get(level.lower(), "badge-low")


def _bar_color(probability: float) -> str:
    if probability >= 0.7:
        return "linear-gradient(90deg, #238636, #2EA043)"
    if probability >= 0.4:
        return "linear-gradient(90deg, #9E6A03, #D29922)"
    return "linear-gradient(90deg, #8B1A1A, #DA3633)"


# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-icon">🔍</div>
    <div>
        <p class="hero-title">Duplicate Question Detector</p>
        <p class="hero-subtitle">
            Determine whether two questions share the same intent using
            Random Forest classification and NLP feature engineering.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------------------------------
# Input Section
# -------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        '<span class="input-label"><span class="input-badge badge-q1">Q1</span>First Question</span>',
        unsafe_allow_html=True,
    )
    q1 = st.text_area(
        label="q1_hidden",
        label_visibility="collapsed",
        placeholder="What is the best way to learn programming?",
        height=130,
        key="q1",
    )

with col2:
    st.markdown(
        '<span class="input-label"><span class="input-badge badge-q2">Q2</span>Second Question</span>',
        unsafe_allow_html=True,
    )
    q2 = st.text_area(
        label="q2_hidden",
        label_visibility="collapsed",
        placeholder="How can I start learning to code effectively?",
        height=130,
        key="q2",
    )

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------
# Analyze Button
# -------------------------------------------------------
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze = st.button("⚡ Analyze Question Pair", use_container_width=True)

# -------------------------------------------------------
# Prediction Logic
# -------------------------------------------------------
if analyze:
    if not q1.strip() or not q2.strip():
        st.warning("⚠️  Please enter both questions before analyzing.")
        st.stop()

    with st.spinner("Running NLP pipeline…"):
        query = create_feature_vector(q1, q2)
        result = predict_duplicate(query)

    probability: float = result["probability"]
    confidence: str    = result["confidence"]
    is_duplicate: bool = result["duplicate"]

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Result Banner ──────────────────────────────────
    if is_duplicate:
        st.markdown(
            '<div class="result-banner result-duplicate">'
            '<span class="result-icon">✅</span>'
            'These questions are <strong>duplicates</strong> — they express the same intent.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-banner result-not-duplicate">'
            '<span class="result-icon">✗</span>'
            'These questions are <strong>not duplicates</strong> — they ask different things.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Metric Cards ───────────────────────────────────
    m1, m2 = st.columns(2, gap="large")

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Duplicate Probability</div>
                <div class="metric-value">{probability:.1%}</div>
                <span class="metric-badge {'badge-high' if probability >= 0.7 else 'badge-medium' if probability >= 0.4 else 'badge-low'}">
                    {'High likelihood' if probability >= 0.7 else 'Moderate likelihood' if probability >= 0.4 else 'Low likelihood'}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-value">{confidence.capitalize()}</div>
                <span class="metric-badge {_confidence_class(confidence)}">
                    {confidence.upper()} CONFIDENCE
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Probability Bar ─────────────────────────────────
    pct = int(probability * 100)
    st.markdown(
        f"""
        <div class="prob-bar-wrap">
            <div class="prob-bar-header">
                <span class="prob-bar-title">Similarity Score</span>
                <span class="prob-bar-pct">{pct} / 100</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{pct}%; background:{_bar_color(probability)};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Raw Payload ─────────────────────────────────────
    with st.expander("📋  Raw prediction payload"):
        st.json(result)

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        <span class="footer-left">Duplicate Question Detector · v1.0</span>
        <div class="footer-pills">
            <span class="pill">Streamlit</span>
            <span class="pill">Random Forest</span>
            <span class="pill">NLP</span>
            <span class="pill">scikit-learn</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)