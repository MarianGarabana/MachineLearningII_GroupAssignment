"""
AI Adoption Stage Classifier — Streamlit App
Group 3 · MBDS 2026 · ML2 Group Assignment · IE University
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="AI Adoption Classifier",
    page_icon="🤖",
)

# ─────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────
ACCENT   = "#2E75B6"
ACCENT2  = "#4FC3F7"
SUCCESS  = "#4CAF50"
WARNING  = "#FF9800"
RED      = "#EF5350"
CARD_BG  = "#1C2333"
GREY     = "#B0BEC5"

STAGE_COLOURS = {"none": RED, "pilot": WARNING, "partial": ACCENT, "full": SUCCESS}
STAGE_LABELS  = {"none": "None", "pilot": "Pilot", "partial": "Partial", "full": "Full"}
STAGE_DESCRIPTIONS = {
    "none":    "The company has not yet begun its AI journey.",
    "pilot":   "The company is in early AI experimentation.",
    "partial": "The company has meaningful AI adoption underway.",
    "full":    "The company is a reference case for AI adoption.",
}
STAGE_RECS = {
    "none":    "This company has not yet begun its AI journey. Priority actions: establish an AI ethics committee, allocate a minimum AI budget, and begin with a low-risk pilot use case in one department.",
    "pilot":   "This company is in early AI experimentation. To progress to partial adoption: increase ai_training_hours, track ai_maturity_score quarterly, and expand the number of active AI projects beyond the current pilot.",
    "partial": "This company has meaningful AI adoption underway. To reach full adoption: target ai_budget_percentage above 15%, reduce ai_failure_rate through better governance, and scale successful use cases organisation-wide.",
    "full":    "This company is a reference case for AI adoption. Recommendations: document and share best practices internally, explore advanced AI governance frameworks, and measure ROI on reskilled employees to sustain competitive advantage.",
}

FEATURE_COLS = [
    "survey_year", "quarter", "country", "region", "industry", "company_size",
    "num_employees", "annual_revenue_usd_millions", "company_age", "company_age_group",
    "years_using_ai", "ai_primary_tool", "num_ai_tools_used", "ai_use_case",
    "ai_projects_active", "ai_training_hours", "ai_budget_percentage", "ai_maturity_score",
    "ai_failure_rate", "ai_investment_per_employee", "regulatory_compliance_score",
    "data_privacy_level", "ai_ethics_committee", "ai_risk_management_score",
    "remote_work_percentage", "employee_satisfaction_score", "task_automation_rate",
    "time_saved_per_week", "productivity_change_percent", "jobs_displaced", "jobs_created",
    "reskilled_employees", "revenue_growth_percent", "cost_reduction_percent",
    "innovation_score", "customer_satisfaction",
]

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0E1117;
    color: #FFFFFF;
    font-family: 'Segoe UI', sans-serif;
}

/* ── tabs ── */
button[data-baseweb="tab"] {
    font-size: 0.95rem;
    font-weight: 600;
    color: #B0BEC5;
    padding: 10px 20px;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #4FC3F7;
    border-bottom: 3px solid #4FC3F7;
}

/* ── custom card ── */
.card {
    background: #1C2333;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

/* ── pipeline step ── */
.pipe-step {
    background: #1C2333;
    border: 1px solid #2E75B6;
    border-radius: 8px;
    padding: 14px 10px;
    text-align: center;
    font-size: 0.82rem;
    color: #B0BEC5;
}
.pipe-icon { font-size: 1.6rem; display:block; margin-bottom:4px; }

/* ── stat card ── */
.stat-card {
    background: #1C2333;
    border-left: 4px solid #2E75B6;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
}
.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #4FC3F7;
    line-height: 1.1;
}
.stat-label {
    font-size: 0.78rem;
    color: #B0BEC5;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── leakage alert ── */
.leakage-alert {
    background: linear-gradient(135deg, #3a1a1a, #2a1010);
    border: 2px solid #EF5350;
    border-radius: 10px;
    padding: 20px 24px;
    margin: 12px 0;
}

/* ── prediction badge ── */
.pred-badge {
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
    margin-bottom: 16px;
}
.pred-stage {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}
.pred-desc { font-size: 0.95rem; color: #B0BEC5; margin-top: 6px; }

/* ── finding card ── */
.finding-card {
    background: #1C2333;
    border-left: 4px solid #4FC3F7;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.finding-num {
    font-size: 1.4rem;
    font-weight: 800;
    color: #4FC3F7;
    margin-right: 10px;
}

/* ── threshold card ── */
.threshold-card {
    background: #1C2333;
    border-radius: 10px;
    padding: 20px 22px;
    height: 100%;
}

/* ── diagnostic callout ── */
.diagnostic-box {
    background: linear-gradient(135deg, #0d2137, #112a42);
    border: 2px solid #4FC3F7;
    border-radius: 12px;
    padding: 24px 28px;
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FILE LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_all():
    file_map = {
        "final_xgb":  "final_xgb.pkl",
        "final_rf":   "final_rf.pkl",
        "tree_cv":    "tree_cv.pkl",
        "cv_rf":      "cvRF.pkl",
        "cv_xgb":     "cv_xgb.pkl",
        "preprocessor": "preprocessor.pkl",
    }
    loaded, missing = {}, []
    for key, fname in file_map.items():
        try:
            loaded[key] = joblib.load(fname)
        except Exception:
            missing.append(fname)
    df_perm = None
    try:
        df_perm = pd.read_csv("df_perm.csv")
    except Exception:
        missing.append("df_perm.csv")
    return loaded, df_perm, missing


models, df_perm, missing_files = load_all()

if missing_files:
    st.markdown("""
<div class="card" style="border:2px solid #FF9800; max-width:720px; margin:40px auto;">
<h2 style="color:#FF9800;">⚙️ Setup required</h2>
<p style="color:#B0BEC5;">The following files are missing. Run the Jupyter notebook to generate them, then place them in the same directory as <code>app.py</code>.</p>
""", unsafe_allow_html=True)
    for f in missing_files:
        st.markdown(f"- `{f}`")
    st.markdown("""
<hr style="border-color:#2E75B6;">
<b>Files expected in this directory:</b>
<ul style="color:#B0BEC5; font-size:0.9rem;">
<li><code>final_xgb.pkl</code> — trained XGBoost classifier</li>
<li><code>final_rf.pkl</code> — trained Random Forest classifier</li>
<li><code>tree_cv.pkl</code> — Decision Tree GridSearchCV object</li>
<li><code>cvRF.pkl</code> — RF GridSearchCV object</li>
<li><code>cv_xgb.pkl</code> — XGBoost GridSearchCV object</li>
<li><code>preprocessor.pkl</code> — fitted ColumnTransformer (save with <code>joblib.dump(preprocessor, 'preprocessor.pkl')</code>)</li>
<li><code>df_perm.csv</code> — permutation importance DataFrame (columns: feature, importance_mean, importance_std)</li>
</ul>
</div>
""", unsafe_allow_html=True)
    st.stop()

# Convenience references
final_xgb   = models["final_xgb"]
final_rf    = models["final_rf"]
tree_cv     = models["tree_cv"]
preprocessor = models["preprocessor"]

# ─────────────────────────────────────────────────────────────
# HELPER: dark Plotly figure defaults
# ─────────────────────────────────────────────────────────────
def dark_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "📊 EDA",
    "🤖 Models",
    "🔮 Live Predictor",
    "💡 Findings & Recommendations",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
<div style="text-align:center; padding: 28px 0 12px 0;">
  <h1 style="font-size:2.6rem; font-weight:800; margin:0;">🤖 AI Adoption Stage Classifier</h1>
  <p style="color:#B0BEC5; font-size:1.05rem; margin-top:6px;">
    Group 3 &nbsp;·&nbsp; MBDS 2026 &nbsp;·&nbsp; ML2 Group Assignment &nbsp;·&nbsp; IE University
  </p>
</div>
""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        with st.container(border=True):
            st.subheader(":material/description: Project summary")
            st.markdown("""
**Business problem**
Predict a company's current AI adoption stage — *none / pilot / partial / full* — from its
operational and financial metrics, enabling consultants to deliver instant, data-driven
diagnostics instead of lengthy manual assessments.

**Dataset**
- 150,000 company records
- 36 features after leakage removal (101 after one-hot encoding)
- 4-class imbalanced target variable

**Metric chosen: F1-macro**
Accuracy is misleading when classes are severely imbalanced. F1-macro gives equal weight
to every class regardless of size — critical here because *full* adopters represent only 1.1 % of records.
""")

    with col_right:
        with st.container(border=True):
            st.subheader(":material/donut_large: Class distribution")
            labels = ["Full (1.1%)", "None (3.5%)", "Partial (52.5%)", "Pilot (42.9%)"]
            values = [1.1, 3.5, 52.5, 42.9]
            colours = [SUCCESS, RED, ACCENT, WARNING]
            donut = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.55,
                marker=dict(colors=colours, line=dict(color="#0E1117", width=2)),
                textinfo="label+percent",
                textfont=dict(size=12),
            ))
            donut.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=260,
            )
            st.plotly_chart(donut, use_container_width=True)
            st.info(
                "**Severe class imbalance:** *partial* dominates at 52.5 %. "
                "A naive classifier always predicting 'partial' scores 52.5 % accuracy — "
                "so we optimise F1-macro instead.",
                icon=":material/warning:",
            )

    # ── Pipeline visual ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(":material/linear_scale: Modelling pipeline")

    steps = [
        ("📥", "Data Loading", "150k records<br>raw CSV"),
        ("🔍", "EDA & Leakage Detection", "Removed ai_adoption_rate<br>Stratified split"),
        ("⚙️", "Preprocessing", "Imputation<br>One-hot encoding"),
        ("🧠", "Model Training", "DT → RF → XGBoost<br>GridSearchCV"),
        ("📈", "Evaluation & SHAP", "F1-macro<br>Permutation importance"),
    ]
    pipe_cols = st.columns(len(steps))
    for col, (icon, title, detail) in zip(pipe_cols, steps):
        with col:
            st.markdown(f"""
<div class="pipe-step">
  <span class="pipe-icon">{icon}</span>
  <strong style="color:#FFFFFF; font-size:0.84rem;">{title}</strong><br>
  <span style="font-size:0.75rem; color:#B0BEC5;">{detail}</span>
</div>
""", unsafe_allow_html=True)

    # ── Stat cards ───────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(":material/leaderboard: Key results")

    stats = [
        ("XGBoost", "Best model"),
        ("0.783", "Best F1-macro (test)"),
        ("1.61 %", "Overfitting gap"),
        ("150,000", "Dataset records"),
    ]
    stat_cols = st.columns(4)
    for col, (val, label) in zip(stat_cols, stats):
        with col:
            st.markdown(f"""
<div class="stat-card">
  <div class="stat-value">{val}</div>
  <div class="stat-label">{label}</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader(":material/analytics: Exploratory data analysis")

    # ── Class imbalance ──────────────────────────────────────
    with st.container(border=True):
        st.markdown("**Class imbalance**")
        class_data = {
            "Stage":  ["None", "Pilot", "Partial", "Full"],
            "Count":  [5198,   64317,   78800,     1685],
            "Colour": [RED,    WARNING, ACCENT,    SUCCESS],
        }
        bar_fig = go.Figure(go.Bar(
            x=class_data["Stage"],
            y=class_data["Count"],
            marker_color=class_data["Colour"],
            text=[f"{c:,}" for c in class_data["Count"]],
            textposition="outside",
        ))
        bar_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="AI adoption stage",
            yaxis_title="Record count",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(bar_fig, use_container_width=True)
        st.caption(
            "A model that always predicts 'partial' would score **52.5 % accuracy** without learning "
            "anything meaningful. This is why accuracy alone is misleading and F1-macro is the right metric."
        )

    # ── Leakage alert ────────────────────────────────────────
    st.markdown(" ", unsafe_allow_html=True)
    st.markdown("""
<div class="leakage-alert">
  <h3 style="color:#EF5350; margin:0 0 10px 0;">🚨 Critical finding: data leakage detected &amp; fixed</h3>
  <p style="color:#FFFFFF; margin:0 0 10px 0;">
    The feature <code>ai_adoption_rate</code> was a <strong>direct numerical encoding of the target variable</strong>
    — non-overlapping ranges mapped perfectly onto each class.
    Including it produced a fake 100 % F1-macro. After removal, honest performance dropped to 74.6 %.
  </p>
</div>
""", unsafe_allow_html=True)

    leak_col1, leak_col2 = st.columns([1, 1.2], gap="large")
    with leak_col1:
        leak_df = pd.DataFrame({
            "AI adoption stage": ["None", "Pilot", "Partial", "Full"],
            "ai_adoption_rate range": ["0 – 9.99", "10 – 34.99", "35 – 69.99", "70 – 100"],
        })
        st.dataframe(leak_df, hide_index=True, use_container_width=True)

    with leak_col2:
        before_after = go.Figure()
        before_after.add_trace(go.Bar(
            name="With leakage (fake)",
            x=["F1-macro"],
            y=[1.0],
            marker_color=RED,
            text=["100 %"],
            textposition="outside",
        ))
        before_after.add_trace(go.Bar(
            name="After fix (honest)",
            x=["F1-macro"],
            y=[0.746],
            marker_color=SUCCESS,
            text=["74.6 %"],
            textposition="outside",
        ))
        before_after.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode="group",
            yaxis=dict(range=[0, 1.15], tickformat=".0%"),
            height=260,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(before_after, use_container_width=True)

    # ── Feature overview ─────────────────────────────────────
    st.markdown(" ", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Feature overview**")
        feat_col1, feat_col2 = st.columns(2)
        with feat_col1:
            st.metric("Numeric features", "26", border=True)
        with feat_col2:
            st.metric("Categorical → after one-hot", "10 → 101", border=True)
        st.markdown("""
**Preprocessing pipeline (applied after stratified split to prevent any leakage):**
- Numeric: **median imputation** for missing values
- Categorical: **mode imputation** then **one-hot encoding** (drop first)
- Stratified 70/30 split applied *before* any fitting — preprocessor fitted on train only
""")

    # ── Train/test split ─────────────────────────────────────
    st.markdown(" ", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Train / test split — stratification check**")
        split_df = pd.DataFrame({
            "Split":   ["Train (105,000)", "Test (45,000)"],
            "None %":  [3.47, 3.47],
            "Pilot %": [42.88, 42.88],
            "Partial %": [52.53, 52.53],
            "Full %":  [1.12, 1.12],
        })
        st.dataframe(split_df, hide_index=True, use_container_width=True)
        st.caption(
            "Stratified split preserves class proportions exactly — both sets have identical "
            "class percentages, confirming no sampling bias."
        )

# ═══════════════════════════════════════════════════════════════
# TAB 3 — MODELS
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.subheader(":material/model_training: Model development & comparison")
    st.markdown("""
A deliberate **3-model progression** was designed to build intuition and justify complexity:
**Decision Tree** (interpretable baseline, exposes overfitting) →
**Random Forest** (parallel ensemble, variance reduction via bootstrap) →
**XGBoost** (sequential ensemble, regularisation via shrinkage).
""")

    # ── Comparison table ─────────────────────────────────────
    st.markdown("#### Model comparison")
    model_df = pd.DataFrame({
        "Model":            ["Decision Tree", "Random Forest", "XGBoost"],
        "CV F1-macro":      [0.749, 0.768, 0.901],
        "Test F1-macro":    [0.742, 0.763, 0.783],
        "Test accuracy":    [0.822, 0.833, 0.862],
        "Recall-macro":     [0.852, 0.880, 0.903],
        "Precision-macro":  [0.720, 0.735, 0.754],
        "Overfitting gap":  ["~25 pp", "Small", "1.61 %"],
    })

    def highlight_xgb(row):
        if row["Model"] == "XGBoost":
            return [f"background-color: {ACCENT}22; color: #4FC3F7; font-weight:700"] * len(row)
        return [""] * len(row)

    styled_df = model_df.style.apply(highlight_xgb, axis=1)
    st.dataframe(styled_df, hide_index=True, use_container_width=True)

    # ── Interactive grouped bar chart ─────────────────────────
    st.markdown("#### Interactive metric comparison")
    metric_opts = ["CV F1-macro", "Test F1-macro", "Test accuracy", "Recall-macro", "Precision-macro"]
    highlight_metric = st.selectbox("Highlight metric", metric_opts, index=1)

    numeric_map = {
        "CV F1-macro":     [0.749, 0.768, 0.901],
        "Test F1-macro":   [0.742, 0.763, 0.783],
        "Test accuracy":   [0.822, 0.833, 0.862],
        "Recall-macro":    [0.852, 0.880, 0.903],
        "Precision-macro": [0.720, 0.735, 0.754],
    }
    model_names = ["Decision Tree", "Random Forest", "XGBoost"]
    bar_colours_map = {m: [ACCENT2 if m == highlight_metric else GREY] * 3 for m in metric_opts}

    grouped_fig = go.Figure()
    for metric in metric_opts:
        opacity = 1.0 if metric == highlight_metric else 0.3
        grouped_fig.add_trace(go.Bar(
            name=metric,
            x=model_names,
            y=numeric_map[metric],
            opacity=opacity,
            marker_color=ACCENT2 if metric == highlight_metric else GREY,
        ))
    grouped_fig = dark_fig(grouped_fig)
    grouped_fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0.6, 1.0]),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(grouped_fig, use_container_width=True)

    # ── Per-model expanders ───────────────────────────────────
    st.markdown("#### Per-model deep dive")

    with st.expander("Decision Tree", icon=":material/account_tree:"):
        col_a, col_b = st.columns([1, 1.5], gap="large")
        with col_a:
            st.markdown("""
**Best params** (GridSearchCV, 36 candidates, 5-fold CV)
- `max_depth = 11`
- `min_samples_leaf = 10`

**CV F1-macro:** 0.749

**Overfitting story:**
Unpruned tree → 100 % train / 74.56 % test.
Pruning via GridSearch reduces the gap meaningfully.
""")
            st.info("Interpretable baseline. Exposes the overfitting problem clearly.", icon=":material/lightbulb:")

        with col_b:
            dt_report = pd.DataFrame({
                "Class":     ["full", "none", "partial", "pilot", "macro avg"],
                "Precision": [0.19, 1.00, 0.87, 0.82, 0.72],
                "Recall":    [0.77, 1.00, 0.78, 0.86, 0.85],
                "F1":        [0.30, 1.00, 0.82, 0.84, 0.74],
                "Support":   [506, 1559, 23640, 19295, "—"],
            })
            st.dataframe(dt_report, hide_index=True, use_container_width=True)

    with st.expander("Random Forest", icon=":material/forest:"):
        col_a, col_b = st.columns([1, 1.5], gap="large")
        with col_a:
            st.markdown("""
**Best params**
- `max_depth = 9`, `max_features = 'sqrt'`, `min_samples_leaf = 10`
- Optimal trees: **90** (OOB curve plateau)
- `class_weight = 'balanced_subsample'` — recomputes weights per bootstrap sample (better than `'balanced'` for RF)
- OOB validation → no separate validation set needed (free from bootstrap)

**Feature reduction experiment:**
Top-15 features → F1 drops from **0.763 to 0.639** (12 pp drop).
Signal is broadly distributed — no small subset captures the full picture.

**Cutoff optimisation:**
OOB probabilities → optimal threshold **0.79** for *full* class.
Precision: 0.24 → **0.45** · Overall F1-macro → 0.80.
""")
        with col_b:
            rf_report = pd.DataFrame({
                "Class":     ["full", "none", "partial", "pilot", "macro avg"],
                "Precision": [0.24, 1.00, 0.87, 0.83, 0.74],
                "Recall":    [0.86, 1.00, 0.80, 0.86, 0.88],
                "F1":        [0.37, 1.00, 0.83, 0.85, 0.76],
                "Support":   [506, 1559, 23640, 19295, "—"],
            })
            st.dataframe(rf_report, hide_index=True, use_container_width=True)

    with st.expander("XGBoost", icon=":material/rocket_launch:"):
        col_a, col_b = st.columns([1, 1.5], gap="large")
        with col_a:
            st.markdown("""
**Best params**
- `max_depth = 5`, `min_child_weight = 50`, `subsample = 1.0`
- `learning_rate = 0.05` (lambda shrinkage), **300 trees**

**CV F1-macro: 0.901** — highest of all three models
**Overfitting gap: 1.61 %** — consistent and robust

**Key distinction from RF:**
Sequential (not parallel) — more trees *can* cause overfitting unlike RF.
Controlled via `learning_rate` and `max_depth`.
""")
            st.success("Best overall model. Highest CV and test F1-macro with minimal overfitting.", icon=":material/trophy:")
        with col_b:
            xgb_report = pd.DataFrame({
                "Class":     ["full", "none", "partial", "pilot", "macro avg"],
                "Precision": [0.25, 1.00, 0.90, 0.87, 0.75],
                "Recall":    [0.89, 1.00, 0.83, 0.89, 0.90],
                "F1":        [0.39, 1.00, 0.86, 0.88, 0.78],
                "Support":   [506, 1559, 23640, 19295, "—"],
            })
            st.dataframe(xgb_report, hide_index=True, use_container_width=True)

    # ── Permutation importance ────────────────────────────────
    st.markdown("#### Top 15 features — permutation importance (F1-macro)")

    if df_perm is not None:
        top15 = df_perm.nlargest(15, "importance_mean").sort_values("importance_mean")
        perm_fig = go.Figure(go.Bar(
            x=top15["importance_mean"],
            y=top15["feature"],
            orientation="h",
            marker_color=ACCENT2,
            error_x=dict(type="data", array=top15["importance_std"], visible=True, color=GREY),
        ))
        perm_fig = dark_fig(perm_fig)
        perm_fig.update_layout(
            xaxis_title="Mean decrease in F1-macro when feature is shuffled",
            height=440,
        )
        st.plotly_chart(perm_fig, use_container_width=True)
        st.caption(
            "Permutation importance is more reliable than impurity-based importance for correlated features — "
            "it measures actual prediction degradation when each feature is shuffled."
        )
    else:
        st.warning("df_perm.csv not found. Run the notebook to generate permutation importances.", icon=":material/warning:")

# ═══════════════════════════════════════════════════════════════
# TAB 4 — LIVE PREDICTOR
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.subheader(":material/model_training: Company AI adoption stage predictor")
    st.caption("Input a company profile and get a real-time prediction from the trained XGBoost model.")

    inp_col, out_col = st.columns([1.1, 1.2], gap="large")

    with inp_col:
        with st.container(border=True):
            st.markdown("**:material/tune: Top predictors**")
            st.caption("The 6 features with highest permutation importance across all models.")

            ai_maturity_score    = st.slider("AI maturity score", 0.0, 1.0, 0.35, 0.01,
                                             help="#1 predictor — composite measure of AI capability maturity")
            years_using_ai       = st.slider("Years using AI", 0, 20, 3, 1,
                                             help="#2 predictor — sustained early investment drives progression")
            ai_budget_percentage = st.slider("AI budget (% of revenue)", 0.0, 30.0, 5.0, 0.1,
                                             help="#3 predictor — long-term commitment signal")
            ai_training_hours    = st.slider("AI training hours per year", 0.0, 500.0, 40.0, 1.0,
                                             help="#4 predictor — directly controllable lever")
            ai_failure_rate      = st.slider("AI failure rate", 0.0, 1.0, 0.3, 0.01,
                                             help="#5 predictor — lower = better governance")
            industry             = st.selectbox("Industry", [
                "Technology", "Finance", "Healthcare", "Education",
                "Retail", "Manufacturing", "Energy", "Other",
            ], help="Sector — Technology and Finance progress fastest")

        # ── All other features hardcoded to sensible defaults ──
        num_employees               = 500
        annual_revenue_usd_millions = 100.0
        employee_satisfaction_score = 6.0
        company_age                 = 15
        survey_year = 2024; quarter = "Q1"; region = "North America"; country = "United States"
        num_ai_tools_used = 3; ai_use_case = "Automation"; ai_projects_active = 5
        ai_primary_tool = "ChatGPT"; regulatory_compliance_score = 70
        data_privacy_level = "Medium"; ai_ethics_committee = "No"
        ai_risk_management_score = 60; remote_work_percentage = 40.0
        task_automation_rate = 0.3; time_saved_per_week = 5.0
        productivity_change_percent = 10.0; jobs_displaced = 10; jobs_created = 8
        reskilled_employees = 5; revenue_growth_percent = 8.0
        cost_reduction_percent = 5.0; innovation_score = 60; customer_satisfaction = 7.0

        predict_btn = st.button(
            "🔮 Predict adoption stage",
            type="primary",
            use_container_width=True,
        )

    with out_col:
        if predict_btn:
            # ── Derive categorical columns ────────────────────
            if num_employees < 50:
                company_size = "Startup"
            elif num_employees < 250:
                company_size = "SME"
            elif num_employees < 1000:
                company_size = "Mid-size"
            else:
                company_size = "Enterprise"

            if company_age < 5:
                company_age_group = "0-5 years"
            elif company_age <= 15:
                company_age_group = "6-15 years"
            elif company_age <= 30:
                company_age_group = "16-30 years"
            else:
                company_age_group = "30+ years"

            ai_investment_per_employee = (annual_revenue_usd_millions * 1000 / max(num_employees, 1)) * 0.05

            input_dict = {
                "survey_year": survey_year,
                "quarter": quarter,
                "country": country,
                "region": region,
                "industry": industry,
                "company_size": company_size,
                "num_employees": num_employees,
                "annual_revenue_usd_millions": annual_revenue_usd_millions,
                "company_age": company_age,
                "company_age_group": company_age_group,
                "years_using_ai": years_using_ai,
                "ai_primary_tool": ai_primary_tool,
                "num_ai_tools_used": num_ai_tools_used,
                "ai_use_case": ai_use_case,
                "ai_projects_active": ai_projects_active,
                "ai_training_hours": ai_training_hours,
                "ai_budget_percentage": ai_budget_percentage,
                "ai_maturity_score": ai_maturity_score,
                "ai_failure_rate": ai_failure_rate,
                "ai_investment_per_employee": ai_investment_per_employee,
                "regulatory_compliance_score": regulatory_compliance_score,
                "data_privacy_level": data_privacy_level,
                "ai_ethics_committee": ai_ethics_committee,
                "ai_risk_management_score": ai_risk_management_score,
                "remote_work_percentage": remote_work_percentage,
                "employee_satisfaction_score": employee_satisfaction_score,
                "task_automation_rate": task_automation_rate,
                "time_saved_per_week": time_saved_per_week,
                "productivity_change_percent": productivity_change_percent,
                "jobs_displaced": jobs_displaced,
                "jobs_created": jobs_created,
                "reskilled_employees": reskilled_employees,
                "revenue_growth_percent": revenue_growth_percent,
                "cost_reduction_percent": cost_reduction_percent,
                "innovation_score": innovation_score,
                "customer_satisfaction": customer_satisfaction,
            }

            try:
                input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
                transformed = preprocessor.transform(input_df)
                pred_raw    = final_xgb.predict(transformed)[0]
                proba       = final_xgb.predict_proba(transformed)[0]
                class_order = list(final_xgb.classes_)

                # Normalise label → string stage name.
                # XGBoost may return numpy.int64 if the target was integer-encoded.
                _stage_names = sorted(STAGE_COLOURS.keys())  # alphabetical: full, none, partial, pilot
                if isinstance(pred_raw, (int, np.integer)):
                    # Check if class_order already contains string stage names
                    if class_order and isinstance(class_order[0], str) and class_order[0] in STAGE_COLOURS:
                        pred_label  = str(pred_raw)
                    else:
                        # Alphabetical LabelEncoder mapping (full=0, none=1, partial=2, pilot=3)
                        _int_to_stage = {i: s for i, s in enumerate(_stage_names)}
                        pred_label  = _int_to_stage.get(int(pred_raw), str(pred_raw))
                        class_order = [_int_to_stage.get(i, str(i)) for i in range(len(class_order))]
                else:
                    pred_label = str(pred_raw)

                st.session_state["prediction_result"] = {
                    "pred_label": pred_label,
                    "proba": proba,
                    "class_order": class_order,
                    "transformed": transformed,
                }
            except Exception as e:
                st.error(f"Prediction failed: {e}", icon=":material/error:")
                st.session_state.pop("prediction_result", None)

        if "prediction_result" in st.session_state:
            res         = st.session_state["prediction_result"]
            pred_label  = res["pred_label"]
            proba       = res["proba"]
            class_order = res["class_order"]
            transformed = res["transformed"]

            # Defensive normalisation — handles stale session state with numpy integer labels
            _stage_names = sorted(STAGE_COLOURS.keys())  # full, none, partial, pilot
            _int_to_stage = {i: s for i, s in enumerate(_stage_names)}
            if isinstance(pred_label, (int, np.integer)):
                pred_label  = _int_to_stage.get(int(pred_label), str(pred_label))
                class_order = [_int_to_stage.get(i, str(c)) for i, c in enumerate(class_order)]
            else:
                pred_label = str(pred_label)

            colour = STAGE_COLOURS.get(pred_label, ACCENT)

            # 1 — Prediction badge
            st.markdown(f"""
<div class="pred-badge" style="background: {colour}22; border: 2px solid {colour};">
  <div class="pred-stage" style="color:{colour};">{pred_label.upper()}</div>
  <div class="pred-desc">{STAGE_DESCRIPTIONS.get(pred_label, '')}</div>
</div>
""", unsafe_allow_html=True)

            # 2 — Confidence chart
            prob_colours = [STAGE_COLOURS.get(c, GREY) if c == pred_label else GREY for c in class_order]
            conf_fig = go.Figure(go.Bar(
                x=proba,
                y=[c.capitalize() for c in class_order],
                orientation="h",
                marker_color=prob_colours,
                text=[f"{p:.1%}" for p in proba],
                textposition="outside",
            ))
            conf_fig = dark_fig(conf_fig)
            conf_fig.update_layout(
                xaxis=dict(range=[0, 1.15], tickformat=".0%"),
                height=220,
                title="Prediction confidence",
            )
            st.plotly_chart(conf_fig, use_container_width=True)

            # 3 — SHAP explanation
            try:
                import shap as shap_lib

                explainer  = shap_lib.TreeExplainer(final_xgb)
                shap_vals  = explainer.shap_values(transformed)

                # Handle both list-of-arrays and 3D array output
                if isinstance(shap_vals, list):
                    pred_idx   = class_order.index(pred_label)
                    sv_for_class = shap_vals[pred_idx][0]
                elif shap_vals.ndim == 3:
                    pred_idx   = class_order.index(pred_label)
                    sv_for_class = shap_vals[0, :, pred_idx]
                else:
                    sv_for_class = shap_vals[0]

                # Feature names from preprocessor
                try:
                    feat_names = preprocessor.get_feature_names_out()
                except Exception:
                    feat_names = [f"f{i}" for i in range(len(sv_for_class))]

                shap_series = pd.Series(sv_for_class, index=feat_names)
                top_shap    = shap_series.abs().nlargest(8).index
                shap_top    = shap_series[top_shap].sort_values()

                shap_colours = [SUCCESS if v >= 0 else RED for v in shap_top.values]
                shap_fig = go.Figure(go.Bar(
                    x=shap_top.values,
                    y=[n.replace("num__", "").replace("cat__", "") for n in shap_top.index],
                    orientation="h",
                    marker_color=shap_colours,
                ))
                shap_fig = dark_fig(shap_fig)
                shap_fig.update_layout(
                    title="Why did the model predict this?",
                    xaxis_title="SHAP value (impact on prediction for this class)",
                    height=320,
                )
                st.plotly_chart(shap_fig, use_container_width=True)
                st.caption("Green bars push the prediction toward this stage. Red bars push against it.")

            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}", icon=":material/info:")

            # 4 — Business recommendation
            st.markdown(f"""
<div class="card" style="border-left: 4px solid {colour};">
  <strong style="color:{colour};">Business recommendation</strong><br><br>
  {STAGE_RECS.get(pred_label, '')}
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 5 — FINDINGS & RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.subheader(":material/lightbulb: Key findings & business recommendations")

    # ── Technical findings ───────────────────────────────────
    st.markdown("#### Technical findings")

    findings = [
        (
            "1",
            "AI maturity score — the dominant predictor",
            "`ai_maturity_score` is the single strongest predictor across all 3 models and both "
            "importance methods (impurity-based and permutation). This is a consistent finding "
            "regardless of algorithm choice.",
        ),
        (
            "2",
            "Sustained early investment drives progression",
            "`years_using_ai` and `ai_budget_percentage` confirm that sustained early investment "
            "is the primary driver of adoption progression — companies that committed early and maintained "
            "budget allocation consistently reach higher stages.",
        ),
        (
            "3",
            "Geography and timing matter",
            "`country_Brazil`, `country_South Africa`, and `quarter_Q1` appeared in the top 15 "
            "permutation importances, confirming that AI adoption is influenced by external market "
            "context, not only internal company decisions.",
        ),
        (
            "4",
            "Signal is broadly distributed — no magic subset",
            "Reducing to top-15 features caused a **12 pp F1-macro drop** (0.763 → 0.639). "
            "No small subset captures the full picture — the model needs the breadth of all 101 features "
            "to perform well.",
        ),
        (
            "5",
            "Class imbalance handled without synthetic data",
            "`class_weight='balanced_subsample'` (not SMOTE) was used. "
            "Cutoff optimisation on OOB probabilities improved *full* class precision from **0.24 → 0.45**, "
            "without introducing synthetic artefacts.",
        ),
    ]

    for num, title, detail in findings:
        st.markdown(f"""
<div class="finding-card">
  <span class="finding-num">{num}</span>
  <strong style="color:#FFFFFF;">{title}</strong><br>
  <span style="color:#B0BEC5; font-size:0.9rem;">{detail}</span>
</div>
""", unsafe_allow_html=True)

    # ── Threshold decision guide ─────────────────────────────
    st.markdown(" ", unsafe_allow_html=True)
    st.markdown("#### Threshold decision guide")
    thr_col1, thr_col2 = st.columns(2, gap="large")

    with thr_col1:
        st.markdown(f"""
<div class="threshold-card" style="border-top: 3px solid {SUCCESS};">
  <h4 style="color:{SUCCESS};">High precision mode (threshold 0.79)</h4>
  <strong>Use when:</strong> benchmarking, case studies, finding confirmed full-adopters<br><br>
  <ul style="color:#B0BEC5; font-size:0.9rem; margin:0; padding-left:18px;">
    <li>Precision: 0.24 → <strong style="color:{SUCCESS};">0.45</strong></li>
    <li>Fewer false positives</li>
    <li>Fewer wasted client meetings</li>
  </ul>
</div>
""", unsafe_allow_html=True)

    with thr_col2:
        st.markdown(f"""
<div class="threshold-card" style="border-top: 3px solid {WARNING};">
  <h4 style="color:{WARNING};">High recall mode (threshold 0.50)</h4>
  <strong>Use when:</strong> outreach campaigns, identifying all potential full adopters<br><br>
  <ul style="color:#B0BEC5; font-size:0.9rem; margin:0; padding-left:18px;">
    <li>Recall: <strong style="color:{WARNING};">0.86</strong></li>
    <li>Maximise detection</li>
    <li>Accept more false positives</li>
  </ul>
</div>
""", unsafe_allow_html=True)

    # ── Manager recommendations ──────────────────────────────
    st.markdown(" ", unsafe_allow_html=True)
    st.markdown("#### Recommendations for managers")

    manager_recs = [
        (":material/trending_up:", "Prioritise ai_maturity_score as the primary quarterly KPI",
         "The single strongest predictor across all models — track it quarterly and set progression targets."),
        (":material/school:", "Invest in ai_training_hours",
         "One of the few directly controllable levers regardless of company size or industry."),
        (":material/payments:", "Maintain ai_budget_percentage above 10 %",
         "Sustained budget allocation signals long-term commitment and is a consistent driver of stage progression."),
        (":material/apartment:", "Prioritise Technology and Finance verticals",
         "These sectors progress fastest — talent retention and AI investment here yields the highest return."),
        (":material/public:", "Adjust intervention strategies by region",
         "Geography and market maturity significantly influence adoption pace — one-size-fits-all approaches underperform."),
    ]

    for icon, title, detail in manager_recs:
        with st.container(border=True):
            left, right = st.columns([0.05, 1])
            with left:
                st.markdown(f"**{icon}**")
            with right:
                st.markdown(f"**{title}**  \n{detail}")

    # ── Model as diagnostic tool ─────────────────────────────
    st.markdown(" ", unsafe_allow_html=True)
    st.markdown("""
<div class="diagnostic-box">
  <h3 style="color:#4FC3F7; margin-top:0;">Model as a diagnostic tool</h3>
  <p style="color:#FFFFFF; font-size:1.0rem; margin:0;">
    This model <strong>replaces lengthy manual consultancy assessments</strong>. Given any company's
    operational metrics, it outputs the current AI adoption stage and — via SHAP values — identifies the
    <strong>specific bottleneck blocking progression to the next stage</strong>. A consultant can assess
    any company profile in under 60 seconds.
  </p>
</div>
""", unsafe_allow_html=True)
