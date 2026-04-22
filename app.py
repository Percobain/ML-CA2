import gradio as gr
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load model ──────────────────────────────────────────────────────────────
artifact = joblib.load("models/student_performance_model.joblib")
model = artifact["model"]
FEATURE_COLS = artifact["feature_columns"]
test_accuracy = artifact.get("test_accuracy", 0)
test_f1 = artifact.get("test_f1", 0)

df = pd.read_csv("data/data.csv")

GOOD_STUDY_HOURS = 6.0
GOOD_ATTENDANCE  = 70.0
GOOD_SLEEP_MIN   = 6.0
GOOD_SLEEP_MAX   = 9.0

# ── Chart style constants (Dark Mode) ─────────────────────────────────────────
BG      = "#0f172a"  # Match panel background
APP_BG  = "#020617"  # Match body background
ACCENT  = "#3b82f6"
RED     = "#ef4444"
GRID    = "#1e293b"
MUTED   = "#64748b"
FG      = "#f8fafc"


def build_features(socioeconomic_score, study_hours, sleep_hours, attendance):
    features = {
        "Socioeconomic Score": socioeconomic_score,
        "Study Hours": study_hours,
        "Sleep Hours": sleep_hours,
        "Attendance (%)": attendance,
        "Study_x_Attendance": study_hours * attendance,
        "Study_x_Socioeconomic": study_hours * socioeconomic_score,
        "Attendance_x_Socioeconomic": attendance * socioeconomic_score,
        "Study_sq": study_hours ** 2,
    }
    return pd.DataFrame([features])[FEATURE_COLS]


def generate_reasoning(prediction, probabilities, study_hours, sleep_hours, attendance, socioeconomic_score):
    fail_prob = probabilities[0] * 100
    pass_prob = probabilities[1] * 100
    is_pass   = prediction == 1

    reasons     = []
    suggestions = []

    if study_hours < GOOD_STUDY_HOURS:
        reasons.append(f"Study time ({study_hours:.1f}h/day) is below the recommended {GOOD_STUDY_HOURS:.0f}h.")
        suggestions.append(f"Increase daily study time to at least {GOOD_STUDY_HOURS:.0f} hours.")
    if attendance < GOOD_ATTENDANCE:
        reasons.append(f"Attendance ({attendance:.0f}%) is below the recommended {GOOD_ATTENDANCE:.0f}%.")
        suggestions.append(f"Aim for at least {GOOD_ATTENDANCE:.0f}% class attendance.")
    if sleep_hours < GOOD_SLEEP_MIN:
        reasons.append(f"Sleep ({sleep_hours:.1f}h) is too low — under {GOOD_SLEEP_MIN:.0f}h hurts cognition.")
        suggestions.append(f"Get at least {GOOD_SLEEP_MIN:.0f} hours of sleep per night.")
    elif sleep_hours > GOOD_SLEEP_MAX:
        reasons.append(f"Sleep ({sleep_hours:.1f}h) is above typical — oversleeping may signal low engagement.")
        suggestions.append(f"Keep sleep between {GOOD_SLEEP_MIN:.0f}–{GOOD_SLEEP_MAX:.0f} hours.")
    if study_hours >= GOOD_STUDY_HOURS and attendance >= GOOD_ATTENDANCE:
        reasons.append("Strong study habits and attendance are working in this student's favour.")
    if socioeconomic_score < 0.4:
        reasons.append(f"Lower socioeconomic score ({socioeconomic_score:.2f}) may correlate with fewer resources.")
        suggestions.append("Seek tutoring, study groups, or institutional support programs.")

    verdict     = "PASS" if is_pass else "FAIL"
    card_class  = "pass-card" if is_pass else "fail-card"
    badge_class = "pass-badge" if is_pass else "fail-badge"
    bar_w       = f"{pass_prob:.1f}"
    confidence  = "High" if abs(pass_prob - 50) > 30 else "Moderate"

    html = f"""
<div class="verdict-wrap {card_class}">
  <div class="verdict-header">
    <span class="{badge_class}">{verdict}</span>
    <span class="confidence">{confidence} confidence</span>
  </div>
  <div class="prob-grid">
    <div class="prob-col">
      <span class="prob-num pass-num">{pass_prob:.1f}%</span>
      <span class="prob-label">Pass</span>
    </div>
    <div class="prob-bar-track">
      <div class="prob-bar-fill" style="width:{bar_w}%"></div>
    </div>
    <div class="prob-col">
      <span class="prob-num fail-num">{fail_prob:.1f}%</span>
      <span class="prob-label">Fail</span>
    </div>
  </div>
</div>
"""

    if reasons:
        html += '<div class="factor-block"><p class="factor-title">Key factors</p>'
        for r in reasons:
            html += f'<p class="factor-item">· {r}</p>'
        html += "</div>"

    if suggestions:
        heading = "How to improve" if not is_pass else "Room to improve"
        html += f'<div class="factor-block"><p class="factor-title">{heading}</p>'
        for s in suggestions:
            html += f'<p class="suggest-item">→ {s}</p>'
        html += "</div>"

    html += f'<p class="meta-line">Accuracy {test_accuracy:.1%} · F1 {test_f1:.1%}</p>'
    return html


def _style_ax(ax):
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)


def make_sensitivity_chart(socioeconomic_score, study_hours, sleep_hours, attendance):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3))
    fig.patch.set_facecolor(BG)

    base = dict(socioeconomic=socioeconomic_score, study=study_hours,
                sleep=sleep_hours, attendance=attendance)

    configs = [
        ("Socioeconomic Score", "socioeconomic", np.linspace(0, 1, 50),    axes[0]),
        ("Study Hours",         "study",         np.linspace(0, 12, 50),   axes[1]),
        ("Sleep Hours",         "sleep",         np.linspace(3, 12, 50),   axes[2]),
        ("Attendance (%)",      "attendance",    np.linspace(20, 100, 50), axes[3]),
    ]

    for label, key, sweep, ax in configs:
        probs = []
        for val in sweep:
            v = dict(base); v[key] = val
            p = model.predict_proba(
                build_features(v["socioeconomic"], v["study"], v["sleep"], v["attendance"])
            )[0][1] * 100
            probs.append(p)

        ax.fill_between(sweep, probs, alpha=0.1, color=ACCENT)
        ax.plot(sweep, probs, color=ACCENT, linewidth=2)
        ax.axhline(50, color=GRID, linewidth=1, linestyle="--")
        ax.axvline(base[key], color=RED, linewidth=1.6, alpha=0.85)
        ax.set_title(label, fontsize=9, fontweight="600", pad=7)
        ax.set_ylabel("Pass %", fontsize=7.5)
        ax.set_ylim(-2, 108)
        _style_ax(ax)

    fig.tight_layout(pad=2.2)
    return fig


def make_feature_chart(socioeconomic_score, study_hours, sleep_hours, attendance):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3))
    fig.patch.set_facecolor(BG)

    features = [
        ("Socioeconomic Score", socioeconomic_score, axes[0]),
        ("Study Hours",         study_hours,         axes[1]),
        ("Sleep Hours",         sleep_hours,         axes[2]),
        ("Attendance (%)",      attendance,          axes[3]),
    ]

    for col_name, user_val, ax in features:
        ax.hist(df[col_name], bins=25, color="#1e293b", edgecolor="none", alpha=0.95)
        ax.axvline(user_val, color=ACCENT, linewidth=2.2)
        ax.set_title(col_name, fontsize=9, fontweight="600", pad=7)
        _, ymax = ax.get_ylim()
        ax.text(user_val, ymax * 0.88, f" {user_val}",
                color=ACCENT, fontsize=8, fontweight="700", va="top")
        _style_ax(ax)

    fig.tight_layout(pad=2.2)
    return fig


def predict(socioeconomic_score, study_hours, sleep_hours, attendance):
    input_df   = build_features(socioeconomic_score, study_hours, sleep_hours, attendance)
    prediction = model.predict(input_df)[0]
    probs      = model.predict_proba(input_df)[0]
    reasoning  = generate_reasoning(prediction, probs, study_hours, sleep_hours,
                                    attendance, socioeconomic_score)
    return (reasoning,
            make_sensitivity_chart(socioeconomic_score, study_hours, sleep_hours, attendance),
            make_feature_chart(socioeconomic_score, study_hours, sleep_hours, attendance))


# ── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #020617 !important;
    font-family: 'DM Sans', ui-sans-serif, sans-serif !important;
}

.app-header {
    padding: 32px 0 24px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 24px;
}
.app-header h1 {
    font-size: 1.55rem; font-weight: 700; color: #f8fafc;
    letter-spacing: -0.02em; margin: 0 0 4px;
}
.app-header p { font-size: 0.85rem; color: #94a3b8; margin: 0; }

.input-panel, .result-panel {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 22px 20px;
}
.result-panel { min-height: 240px; }

.panel-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #64748b; margin: 0 0 14px;
}

.predict-btn button {
    background: #3b82f6 !important;
    border: none !important; border-radius: 9px !important;
    font-size: 0.88rem !important; font-weight: 600 !important;
    padding: 11px 0 !important; margin-top: 6px !important;
    box-shadow: 0 4px 14px rgba(59,130,246,.3) !important;
    transition: background .18s, transform .12s !important;
}
.predict-btn button:hover {
    background: #2563eb !important; transform: translateY(-1px) !important;
}

.verdict-wrap {
    border-radius: 10px; padding: 16px 18px;
    margin-bottom: 16px; border: 1px solid transparent;
}
.pass-card { background: rgba(22, 163, 74, 0.1); border-color: rgba(22, 163, 74, 0.2); }
.fail-card { background: rgba(220, 38, 38, 0.1); border-color: rgba(220, 38, 38, 0.2); }

.verdict-header {
    display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 12px;
}
.pass-badge, .fail-badge {
    font-size: 0.95rem; font-weight: 700; letter-spacing: 0.05em;
}
.pass-badge { color: #4ade80; }
.fail-badge { color: #f87171; }
.confidence { font-size: 0.73rem; color: #64748b; font-weight: 500; }

.prob-grid { display: flex; align-items: center; gap: 10px; }
.prob-col { display: flex; flex-direction: column; align-items: center; min-width: 48px; }
.prob-num { font-size: 1rem; font-weight: 700; line-height: 1; }
.pass-num { color: #4ade80; }
.fail-num { color: #f87171; }
.prob-label { font-size: 0.7rem; color: #64748b; margin-top: 2px; }
.prob-bar-track {
    flex: 1; height: 7px; background: #1e293b;
    border-radius: 99px; overflow: hidden;
}
.prob-bar-fill {
    height: 100%; background: #3b82f6;
    border-radius: 99px;
}

.factor-block { margin-bottom: 12px; }
.factor-title {
    font-size: 0.68rem !important; font-weight: 700 !important;
    letter-spacing: 0.09em !important; text-transform: uppercase !important;
    color: #64748b !important; margin: 0 0 6px !important;
}
.factor-item {
    font-size: 0.83rem !important; color: #cbd5e1 !important;
    line-height: 1.5 !important; margin: 0 0 3px !important;
}
.suggest-item {
    font-size: 0.83rem !important; color: #60a5fa !important;
    line-height: 1.5 !important; margin: 0 0 3px !important;
}
.meta-line {
    font-size: 0.72rem !important; color: #475569 !important;
    margin-top: 14px !important; padding-top: 10px !important;
    border-top: 1px solid #1e293b !important;
}
.placeholder {
    color: #475569 !important; font-size: 0.85rem !important;
    padding: 12px 0 !important;
}

.tab-nav button {
    font-size: 0.82rem !important; font-weight: 600 !important; color: #64748b !important;
}
.tab-nav button.selected {
    color: #3b82f6 !important; border-bottom: 2px solid #3b82f6 !important;
}

.chart-caption p {
    font-size: 0.76rem !important; color: #64748b !important;
    text-align: center !important; margin-top: 2px !important;
}

.app-footer {
    text-align: center; padding: 18px 0 4px;
    border-top: 1px solid #1e293b; margin-top: 4px;
    font-size: 0.76rem; color: #475569;
}

input[type=range] { accent-color: #3b82f6 !important; }

.gradio-plot > div {
    border-radius: 10px !important;
    border: 1px solid #1e293b !important;
    overflow: hidden !important;
    background: #0f172a !important;
}
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill="#020617",
    block_background_fill="#0f172a",
    block_border_width="1px",
    block_border_color="#1e293b",
    block_radius="12px",
    block_shadow="none",
    input_background_fill="#1e293b",
    input_border_color="#1e293b",
    slider_color="#3b82f6",
    button_primary_background_fill="#3b82f6",
    button_primary_background_fill_hover="#2563eb",
    button_primary_text_color="#ffffff",
    button_primary_border_color="transparent",
)

# ── Layout ───────────────────────────────────────────────────────────────────
with gr.Blocks(title="Student Performance Predictor", theme=THEME, css=CSS) as demo:

    gr.HTML("""
    <div class="app-header">
        <h1>Student Performance Predictor</h1>
        <p>Pass / fail prediction from study habits &amp; background · Decision tree · 1,388 records</p>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=280):
            with gr.Column(elem_classes="input-panel"):
                gr.HTML('<p class="panel-label">Student Profile</p>')
                socioeconomic = gr.Slider(0.0, 1.0,   value=0.5,  step=0.01, label="Socioeconomic Score",
                                          info="Normalised 0–1 index of family resources")
                study_hours   = gr.Slider(0.0, 15.0,  value=5.0,  step=0.1,  label="Study Hours / Day",
                                          info="6h+ recommended")
                sleep_hours   = gr.Slider(0.0, 14.0,  value=7.0,  step=0.1,  label="Sleep Hours / Day",
                                          info="6–9h recommended")
                attendance    = gr.Slider(0.0, 100.0, value=65.0, step=1.0,  label="Attendance (%)",
                                          info="70%+ recommended")
                predict_btn   = gr.Button("Analyze", variant="primary", size="lg",
                                          elem_classes="predict-btn")

        with gr.Column(scale=6, min_width=320):
            with gr.Column(elem_classes="result-panel"):
                gr.HTML('<p class="panel-label">Prediction</p>')
                reasoning_output = gr.HTML(
                    '<p class="placeholder">Enter a student profile and click Analyze.</p>'
                )

    with gr.Tabs():
        with gr.TabItem("Sensitivity Analysis"):
            sens_plot = gr.Plot(show_label=False)
            gr.HTML('<div class="chart-caption"><p>How each factor shifts pass probability independently. Red line = current value.</p></div>')
        with gr.TabItem("Population Context"):
            dist_plot = gr.Plot(show_label=False)
            gr.HTML('<div class="chart-caption"><p>Where this student sits in the dataset. Blue line = current value.</p></div>')

    gr.Examples(
        examples=[
            [0.85, 8.0, 7.5, 85],
            [0.30, 2.0, 6.0, 40],
            [0.60, 5.0, 8.0, 70],
            [0.95, 9.5, 7.0, 90],
            [0.20, 1.5, 5.0, 30],
        ],
        inputs=[socioeconomic, study_hours, sleep_hours, attendance],
        outputs=[reasoning_output, sens_plot, dist_plot],
        fn=predict,
        cache_examples=False,
        label="Sample Profiles",
    )

    gr.HTML(f"""
    <div class="app-footer">
        Accuracy {test_accuracy:.1%} · F1 {test_f1:.1%} · Built with Gradio<br>
        Predictions are based on historical patterns and intended as guidance only.
    </div>
    """)

    predict_btn.click(
        fn=predict,
        inputs=[socioeconomic, study_hours, sleep_hours, attendance],
        outputs=[reasoning_output, sens_plot, dist_plot],
    )

if __name__ == "__main__":
    demo.launch()
