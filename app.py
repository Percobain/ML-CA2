import gradio as gr
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load model artifact
artifact = joblib.load("models/student_performance_model.joblib")
model = artifact["model"]
FEATURE_COLS = artifact["feature_columns"]
test_accuracy = artifact.get("test_accuracy", 0)
test_f1 = artifact.get("test_f1", 0)

# Load dataset for distribution charts
df = pd.read_csv("data/data.csv")

GOOD_STUDY_HOURS = 6.0
GOOD_ATTENDANCE = 70.0
GOOD_SLEEP_MIN = 6.0
GOOD_SLEEP_MAX = 9.0


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
    is_pass = prediction == 1

    reasons = []
    suggestions = []

    if study_hours < GOOD_STUDY_HOURS:
        reasons.append(f"Study hours ({study_hours:.1f}h) are below the recommended {GOOD_STUDY_HOURS:.0f}h per day.")
        suggestions.append(f"Increase daily study time to at least {GOOD_STUDY_HOURS:.0f} hours.")

    if attendance < GOOD_ATTENDANCE:
        reasons.append(f"Attendance ({attendance:.0f}%) is below the recommended {GOOD_ATTENDANCE:.0f}%.")
        suggestions.append(f"Aim for at least {GOOD_ATTENDANCE:.0f}% class attendance.")

    if sleep_hours < GOOD_SLEEP_MIN:
        reasons.append(f"Sleep ({sleep_hours:.1f}h) is too low. Less than {GOOD_SLEEP_MIN:.0f}h affects cognitive performance.")
        suggestions.append(f"Get at least {GOOD_SLEEP_MIN:.0f} hours of sleep per night.")
    elif sleep_hours > GOOD_SLEEP_MAX:
        reasons.append(f"Sleep ({sleep_hours:.1f}h) is higher than typical. Oversleeping may indicate low engagement.")
        suggestions.append(f"Keep sleep between {GOOD_SLEEP_MIN:.0f} and {GOOD_SLEEP_MAX:.0f} hours.")

    if study_hours >= GOOD_STUDY_HOURS and attendance >= GOOD_ATTENDANCE:
        reasons.append("Strong study habits and attendance are working in this student's favor.")

    if socioeconomic_score < 0.4:
        reasons.append(f"Lower socioeconomic score ({socioeconomic_score:.2f}) can correlate with fewer resources.")
        suggestions.append("Seek out tutoring, study groups, or institutional support programs.")

    lines = []
    result_label = "PASS" if is_pass else "FAIL"
    lines.append(f"## Prediction: {result_label}")
    lines.append("")
    lines.append(f"Pass: **{pass_prob:.1f}%** | Fail: **{fail_prob:.1f}%**")
    lines.append("")

    lines.append("### Why this prediction")
    if not reasons:
        lines.append("All input factors are within healthy ranges.")
    else:
        for r in reasons:
            lines.append(f"- {r}")

    if suggestions:
        lines.append("")
        heading = "### How to improve" if not is_pass else "### Room for improvement"
        lines.append(heading)
        for s in suggestions:
            lines.append(f"- {s}")

    lines.append("")
    lines.append(f"<sub>Model accuracy: {test_accuracy:.1%} | F1: {test_f1:.1%}</sub>")

    return "\n".join(lines)


def make_sensitivity_chart(socioeconomic_score, study_hours, sleep_hours, attendance):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))

    base_vals = {
        "socioeconomic": socioeconomic_score,
        "study": study_hours,
        "sleep": sleep_hours,
        "attendance": attendance,
    }

    configs = [
        ("Socioeconomic Score", "socioeconomic", np.linspace(0, 1, 40), axes[0]),
        ("Study Hours", "study", np.linspace(0, 12, 40), axes[1]),
        ("Sleep Hours", "sleep", np.linspace(3, 12, 40), axes[2]),
        ("Attendance (%)", "attendance", np.linspace(20, 100, 40), axes[3]),
    ]

    for label, key, sweep, ax in configs:
        probs = []
        for val in sweep:
            vals = dict(base_vals)
            vals[key] = val
            input_df = build_features(vals["socioeconomic"], vals["study"], vals["sleep"], vals["attendance"])
            p = model.predict_proba(input_df)[0][1] * 100
            probs.append(p)

        ax.fill_between(sweep, probs, alpha=0.15, color="#2563eb")
        ax.plot(sweep, probs, color="#2563eb", linewidth=1.8)
        ax.axhline(y=50, color="#999", linewidth=0.7, linestyle="--")
        ax.axvline(x=base_vals[key], color="#e53e3e", linewidth=1.5, linestyle="-", alpha=0.7)
        ax.set_title(label, fontsize=10, fontweight="500")
        ax.set_ylabel("Pass %", fontsize=8)
        ax.set_ylim(-2, 105)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2.0)
    return fig


def make_feature_chart(socioeconomic_score, study_hours, sleep_hours, attendance):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))

    features = [
        ("Socioeconomic Score", socioeconomic_score, axes[0]),
        ("Study Hours", study_hours, axes[1]),
        ("Sleep Hours", sleep_hours, axes[2]),
        ("Attendance (%)", attendance, axes[3]),
    ]

    for col_name, user_val, ax in features:
        ax.hist(df[col_name], bins=25, color="#cbd5e1", edgecolor="none", alpha=0.8)
        ax.axvline(x=user_val, color="#2563eb", linewidth=2, linestyle="-")
        ax.set_title(col_name, fontsize=10, fontweight="500")
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ymin, ymax = ax.get_ylim()
        ax.text(user_val, ymax * 0.92, f" {user_val}", color="#2563eb",
                fontsize=8, fontweight="bold", va="top")

    fig.tight_layout(pad=2.0)
    return fig


def predict(socioeconomic_score, study_hours, sleep_hours, attendance):
    input_df = build_features(socioeconomic_score, study_hours, sleep_hours, attendance)
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    reasoning = generate_reasoning(
        prediction, probabilities,
        study_hours, sleep_hours, attendance, socioeconomic_score
    )
    sensitivity_chart = make_sensitivity_chart(socioeconomic_score, study_hours, sleep_hours, attendance)
    feature_chart = make_feature_chart(socioeconomic_score, study_hours, sleep_hours, attendance)

    return reasoning, sensitivity_chart, feature_chart


with gr.Blocks(
    title="Student Performance Predictor",
    theme=gr.themes.Base(),
    css="""
        .header-text h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 0; }
        .header-text p { color: #666; font-size: 0.85rem; margin-top: 4px; }
        .result-box { min-height: 200px; }
    """,
) as demo:
    gr.Markdown(
        "# Student Performance Predictor\n"
        "Predict pass or fail from student profile. "
        "Decision tree trained on 1388 records.",
        elem_classes="header-text",
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=260):
            socioeconomic = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                label="Socioeconomic Score",
            )
            study_hours = gr.Slider(
                minimum=0.0, maximum=15.0, value=5.0, step=0.1,
                label="Study Hours per Day",
            )
            sleep_hours = gr.Slider(
                minimum=0.0, maximum=14.0, value=7.0, step=0.1,
                label="Sleep Hours per Day",
            )
            attendance = gr.Slider(
                minimum=0.0, maximum=100.0, value=65.0, step=1.0,
                label="Attendance (%)",
            )
            predict_btn = gr.Button("Predict", variant="primary")

        with gr.Column(scale=2, min_width=360):
            reasoning_output = gr.Markdown(
                elem_classes="result-box",
            )

    sens_plot = gr.Plot(label="Sensitivity: how each factor affects pass probability (red line = your value)")
    dist_plot = gr.Plot(label="Where you stand in the dataset (blue line = your value)")

    predict_btn.click(
        fn=predict,
        inputs=[socioeconomic, study_hours, sleep_hours, attendance],
        outputs=[reasoning_output, sens_plot, dist_plot],
    )

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
    )

if __name__ == "__main__":
    demo.launch()
