
import os
import joblib
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from lime.lime_tabular import LimeTabularExplainer

# -----------------------------
# 1. LOAD TRAINED MODEL
# -----------------------------

BUNDLE_PATH = "student_xai_bundle.joblib"

bundle = joblib.load(BUNDLE_PATH)

student_model = bundle["model"]
X_train = bundle["X_train"]

preprocessor = student_model.named_steps["preprocessor"]
classifier = student_model.named_steps["classifier"]

feature_names = list(preprocessor.get_feature_names_out())

# -----------------------------
# 2. CREATE SHAP AND LIME
# -----------------------------

X_train_encoded = preprocessor.transform(X_train)

if hasattr(X_train_encoded, "toarray"):
    X_train_encoded = X_train_encoded.toarray()

background = X_train_encoded[:40]

shap_explainer = shap.TreeExplainer(
    classifier,
    data=background
)

lime_explainer = LimeTabularExplainer(
    training_data=np.asarray(X_train_encoded),
    feature_names=feature_names,
    class_names=["Fail", "Pass"],
    mode="classification",
    random_state=42
)

# -----------------------------
# 3. PREDICTION FUNCTION
# -----------------------------

def explain_student(
    school, sex, age, address, studytime,
    failures, schoolsup, famsup, internet, absences
):

    student = pd.DataFrame([{
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "internet": internet,
        "absences": absences
    }])

    prediction = student_model.predict(student)[0]
    probabilities = student_model.predict_proba(student)[0]

    fail_probability = float(probabilities[0]) * 100
    pass_probability = float(probabilities[1]) * 100

    # Experimental model-based risk indicator
    if fail_probability >= 60:
        risk = "HIGH"
        advice = (
            "Consider reviewing study habits, attendance and previous "
            "class failures. Seek academic support where needed."
        )
    elif fail_probability >= 40:
        risk = "MODERATE"
        advice = (
            "Maintain consistent study routines and monitor attendance "
            "and academic progress."
        )
    else:
        risk = "LOW"
        advice = (
            "Continue maintaining regular study habits and attendance."
        )

    result_text = f"""
## Prediction: {"PASS" if prediction == 1 else "FAIL"}

| Model output | Probability |
|---|---:|
| Pass | {pass_probability:.2f}% |
| Fail | {fail_probability:.2f}% |

### Experimental risk indicator: {risk}

**Suggestions:** {advice}

*This is a model-based estimate, not a guaranteed academic outcome
or a validated clinical risk score.*
"""

    # -----------------------------
    # 4. SHAP EXPLANATION
    # -----------------------------

    encoded_student = preprocessor.transform(student)

    if hasattr(encoded_student, "toarray"):
        encoded_student = encoded_student.toarray()

    shap_result = shap_explainer(encoded_student)
    shap_values = np.asarray(shap_result.values)

    if shap_values.ndim == 3:
        values = shap_values[0, :, 1]
    elif shap_values.ndim == 2:
        values = shap_values[0]
    else:
        values = shap_values.reshape(-1)

    top_indices = np.argsort(np.abs(values))[-10:]

    fig_shap, ax_shap = plt.subplots(figsize=(9, 5))
    ax_shap.barh(
        [feature_names[i] for i in top_indices],
        values[top_indices]
    )
    ax_shap.set_title("SHAP Feature Contributions (Pass Class)")
    ax_shap.set_xlabel("SHAP value")
    fig_shap.tight_layout()

    # -----------------------------
    # 5. LIME EXPLANATION
    # -----------------------------

    lime_result = lime_explainer.explain_instance(
        np.asarray(encoded_student)[0],
        classifier.predict_proba,
        num_features=10,
        labels=[1]
    )

    lime_items = lime_result.as_list(label=1)

    lime_labels = [item[0] for item in lime_items]
    lime_weights = [item[1] for item in lime_items]

    fig_lime, ax_lime = plt.subplots(figsize=(9, 5))
    ax_lime.barh(lime_labels, lime_weights)
    ax_lime.set_title("LIME Feature Contributions (Pass Class)")
    ax_lime.set_xlabel("LIME weight")
    fig_lime.tight_layout()

    return result_text, fig_shap, fig_lime


# -----------------------------
# 6. MODEL PERFORMANCE
# -----------------------------

accuracy = bundle["accuracy"]
precision = bundle["precision"]
recall = bundle["recall"]
f1 = bundle["f1"]

performance_text = f"""
## Random Forest Model Performance

| Metric | Score |
|---|---:|
| Accuracy | {accuracy * 100:.2f}% |
| Pass Precision | {precision * 100:.2f}% |
| Pass Recall | {recall * 100:.2f}% |
| Pass F1 Score | {f1 * 100:.2f}% |

**Training samples:** {len(X_train)}

The model uses a stratified train-test split and excludes G1, G2
and G3 from the input features to avoid target leakage.

*Performance is limited and should be improved before real-world use.*
"""


# -----------------------------
# 7. GRADIO DASHBOARD
# -----------------------------

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Student Performance XAI Dashboard"
) as demo:

    gr.Markdown("""
    # Student Performance Prediction using Explainable AI
    ### A Comparative Study of SHAP and LIME

    Predict student performance and understand the model's decisions.
    """)

    with gr.Tab("Predict & Explain"):

        with gr.Row():

            with gr.Column():

                school = gr.Dropdown(
                    choices=["GP", "MS"],
                    value="GP",
                    label="School"
                )

                sex = gr.Dropdown(
                    choices=["F", "M"],
                    value="F",
                    label="Sex"
                )

                age = gr.Slider(
                    15, 22, value=17, step=1,
                    label="Age"
                )

                address = gr.Dropdown(
                    choices=["U", "R"],
                    value="U",
                    label="Home Address"
                )

                studytime = gr.Dropdown(
                    choices=[
                        ("Less than 2 hours", 1),
                        ("2 to 5 hours", 2),
                        ("5 to 10 hours", 3),
                        ("More than 10 hours", 4)
                    ],
                    value=2,
                    label="Weekly Study Time"
                )

            with gr.Column():

                failures = gr.Slider(
                    0, 4, value=0, step=1,
                    label="Past Class Failures"
                )

                schoolsup = gr.Dropdown(
                    choices=["yes", "no"],
                    value="no",
                    label="School Support"
                )

                famsup = gr.Dropdown(
                    choices=["yes", "no"],
                    value="yes",
                    label="Family Support"
                )

                internet = gr.Dropdown(
                    choices=["yes", "no"],
                    value="yes",
                    label="Internet Access"
                )

                absences = gr.Slider(
                    0, 75, value=2, step=1,
                    label="Number of Absences"
                )

        predict_button = gr.Button(
            "Predict and Explain",
            variant="primary"
        )

        result_output = gr.Markdown()

        with gr.Row():
            shap_output = gr.Plot(label="SHAP Explanation")
            lime_output = gr.Plot(label="LIME Explanation")

        predict_button.click(
            fn=explain_student,
            inputs=[
                school, sex, age, address, studytime,
                failures, schoolsup, famsup, internet, absences
            ],
            outputs=[
                result_output, shap_output, lime_output
            ]
        )

    with gr.Tab("Model Performance"):
        gr.Markdown(performance_text)

    with gr.Tab("About the Project"):
        gr.Markdown("""
        ## Project Overview

        This project predicts student pass/fail outcomes using a
        Random Forest classifier.

        ## Explainable AI Methods

        - **SHAP:** Explains feature contributions to model predictions.
        - **LIME:** Explains individual predictions using a local
          interpretable approximation.

        ## Dataset

        UCI Student Performance — Mathematics dataset.

        ## Important Limitation

        The model is an academic prototype and should not be used
        as the sole basis for decisions about students.
        """)


# -----------------------------
# 8. LAUNCH APPLICATION
# -----------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
