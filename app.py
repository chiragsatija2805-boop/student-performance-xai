import os, joblib, gradio as gr, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, 'student_xai_bundle.joblib')
RESEARCH_PATH = os.path.join(BASE, 'student_performance_research_bundle.joblib')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError('Keep student_xai_bundle.joblib beside app.py in the repository.')
bundle = joblib.load(MODEL_PATH)
model = bundle['model']
X_train = bundle['X_train']
preprocessor = model.named_steps['preprocessor']
classifier = model.named_steps['classifier']
feature_names = list(preprocessor.get_feature_names_out())
X_enc = preprocessor.transform(X_train)
if hasattr(X_enc, 'toarray'): X_enc = X_enc.toarray()
X_enc = np.asarray(X_enc)
background = X_enc[:min(40, len(X_enc))]
shap_explainer = shap.TreeExplainer(classifier, data=background)
lime_explainer = LimeTabularExplainer(X_enc, feature_names=feature_names,
    class_names=['Fail', 'Pass'], mode='classification', random_state=42)

COLS = ['school','sex','age','address','studytime','failures','schoolsup','famsup','internet','absences']
def make_student(school, sex, age, address, studytime, failures, schoolsup, famsup, internet, absences):
    return pd.DataFrame([dict(school=school, sex=sex, age=int(age), address=address,
        studytime=int(studytime), failures=int(failures), schoolsup=schoolsup,
        famsup=famsup, internet=internet, absences=int(absences))], columns=COLS)
def encode(student):
    x = preprocessor.transform(student)
    if hasattr(x, 'toarray'): x = x.toarray()
    return np.asarray(x)
def fail_index(estimator): return list(estimator.classes_).index(0)
def risk_advice(p):
    if p >= 60: return 'HIGH', 'Review attendance, study habits and previous failures; consider academic support.'
    if p >= 40: return 'MODERATE', 'Monitor academic progress and encourage a consistent study routine.'
    return 'LOW', 'Continue regular study habits and attendance.'
def explain_student(school, sex, age, address, studytime, failures, schoolsup, famsup, internet, absences):
    student = make_student(school, sex, age, address, studytime, failures, schoolsup, famsup, internet, absences)
    probs = model.predict_proba(student)[0]; fi = fail_index(model); pi = list(model.classes_).index(1)
    fail_p, pass_p = float(probs[fi])*100, float(probs[pi])*100
    risk, advice = risk_advice(fail_p)
    result = f'''## Prediction: {"PASS" if int(model.predict(student)[0]) == 1 else "FAIL"}

| Model output | Probability |
|---|---:|
| Pass | **{pass_p:.2f}%** |
| Fail | **{fail_p:.2f}%** |

### Experimental risk indicator: {risk}

**Suggested action:** {advice}

*Experimental estimate only; not a guaranteed academic outcome or validated risk score.*'''
    encoded = encode(student)
    sv = np.asarray(shap_explainer(encoded, check_additivity=False).values)
    if sv.ndim == 3: values = sv[0, :, 1]
    elif sv.ndim == 2: values = sv[0]
    else: values = sv.reshape(-1)
    idx = np.argsort(np.abs(values))[-10:]
    fig1, ax1 = plt.subplots(figsize=(8,5)); ax1.barh([feature_names[i] for i in idx], values[idx])
    ax1.axvline(0, linewidth=.8); ax1.set_title('SHAP: Contributions to Pass Prediction')
    ax1.set_xlabel('SHAP value (positive supports Pass)'); fig1.tight_layout()
    lime = lime_explainer.explain_instance(encoded[0], classifier.predict_proba, num_features=10, labels=[1])
    items = lime.as_list(label=1)
    fig2, ax2 = plt.subplots(figsize=(8,5)); ax2.barh([x[0] for x in items], [x[1] for x in items])
    ax2.axvline(0, linewidth=.8); ax2.set_title('LIME: Local Explanation for Pass Class')
    ax2.set_xlabel('LIME weight (positive supports Pass)'); fig2.tight_layout()
    return result, fig1, fig2

def what_if(school, sex, age, address, studytime, failures, schoolsup, famsup, internet, absences):
    student = make_student(school, sex, age, address, studytime, failures, schoolsup, famsup, internet, absences)
    fi = fail_index(model); current = float(model.predict_proba(student)[0][fi])*100
    cases = [('Study time +1 category','studytime',min(int(studytime)+1,4)),
             ('Absences reduced by 5','absences',max(int(absences)-5,0)),
             ('Previous failures set to 0','failures',0)]
    rows=[]
    for label, col, val in cases:
        changed=student.copy(); changed.loc[0,col]=val
        p=float(model.predict_proba(changed)[0][fi])*100
        rows.append({'Scenario':label,'Predicted Fail probability':f'{p:.2f}%','Change vs current':f'{p-current:+.2f} percentage points'})
    note=(f'**Current predicted Fail probability: {current:.2f}%**\n\n'
          'One input is changed at a time. These are model sensitivity checks, not causal effects or guaranteed outcomes.')
    return note, pd.DataFrame(rows)

research=None
if os.path.exists(RESEARCH_PATH):
    try: research=joblib.load(RESEARCH_PATH)
    except Exception as e: print('Optional research bundle could not be loaded:', e)
def comparison_data():
    if research is None:
        return None, None, ('## Six-Algorithm Comparison\n\nThe comparison bundle is not in the repository yet. Add '
            '`student_performance_research_bundle.joblib` beside `app.py` and redeploy. '
            'No comparison scores are fabricated.')
    df=research.get('comparison_table')
    if df is None: return None,None,'Research bundle loaded, but comparison_table is missing.'
    if not isinstance(df,pd.DataFrame): df=pd.DataFrame(df)
    numeric=df.select_dtypes(include=[np.number]).columns.tolist()
    name=df.columns[0] if len(df.columns) else None
    score=next((c for c in ['CV_Fail_F1','CV Fail F1','cv_fail_f1','CV_F1_Fail','Fail_F1'] if c in df.columns), None)
    if score is None and numeric: score=numeric[0]
    fig=None
    if name is not None and score is not None:
        plot=df[[name,score]].copy(); plot[score]=pd.to_numeric(plot[score],errors='coerce'); plot=plot.dropna().sort_values(score)
        fig,ax=plt.subplots(figsize=(9,5)); ax.barh(plot[name].astype(str),plot[score]); ax.set_title(f'Algorithm Comparison: {score}'); ax.set_xlabel('Score'); fig.tight_layout()
    return df,fig,'Results loaded from the saved Colab experiment. Use cross-validation metrics for model selection; test metrics are for final evaluation.'

accuracy=float(bundle.get('accuracy',0)); precision=float(bundle.get('precision',0)); recall=float(bundle.get('recall',0)); f1=float(bundle.get('f1',0))
performance=f'''## Random Forest Model Performance

| Metric | Score |
|---|---:|
| Accuracy | {accuracy*100:.2f}% |
| Pass Precision | {precision*100:.2f}% |
| Pass Recall | {recall*100:.2f}% |
| Pass F1 Score | {f1*100:.2f}% |

**Training samples:** {len(X_train)}

G1, G2 and G3 are excluded from input features to avoid target leakage.

*These are the original saved Random Forest metrics. Performance is limited and should be improved before real-world use.*'''

with gr.Blocks(theme=gr.themes.Soft(primary_hue='violet',secondary_hue='blue'), title='Student Performance XAI Dashboard') as demo:
    gr.Markdown('# 🎓 Student Performance Prediction using Explainable AI\n### A Comparative Study of SHAP and LIME\nPredict student outcomes, compare algorithms and understand model decisions.')
    with gr.Tabs():
        with gr.Tab('🔮 Predict & Explain'):
            gr.Markdown('Enter student information to generate a prediction with SHAP and LIME explanations.')
            with gr.Row():
                with gr.Column():
                    school=gr.Dropdown(['GP','MS'],value='GP',label='School'); sex=gr.Dropdown(['F','M'],value='F',label='Sex')
                    age=gr.Slider(15,22,value=17,step=1,label='Age'); address=gr.Dropdown(['U','R'],value='U',label='Home Address')
                    studytime=gr.Dropdown(choices=[('Less than 2 hours',1),('2 to 5 hours',2),('5 to 10 hours',3),('More than 10 hours',4)],value=2,label='Weekly Study Time')
                with gr.Column():
                    failures=gr.Slider(0,4,value=0,step=1,label='Past Class Failures'); schoolsup=gr.Dropdown(['yes','no'],value='no',label='School Support')
                    famsup=gr.Dropdown(['yes','no'],value='yes',label='Family Support'); internet=gr.Dropdown(['yes','no'],value='yes',label='Internet Access')
                    absences=gr.Slider(0,75,value=2,step=1,label='Number of Absences')
            predict=gr.Button('Predict and Explain',variant='primary'); result=gr.Markdown()
            with gr.Row(): shap_plot=gr.Plot(label='SHAP Local Explanation'); lime_plot=gr.Plot(label='LIME Local Explanation')
            inputs=[school,sex,age,address,studytime,failures,schoolsup,famsup,internet,absences]
            predict.click(explain_student,inputs=inputs,outputs=[result,shap_plot,lime_plot])
        with gr.Tab('📊 Algorithm Comparison'):
            gr.Markdown('Six algorithms: Logistic Regression, Decision Tree, Random Forest, SVM, KNN and Gradient Boosting.')
            df,fig,note=comparison_data(); gr.Markdown(note)
            if df is not None: gr.Dataframe(value=df,interactive=False,label='Saved experiment results')
            if fig is not None: gr.Plot(value=fig,label='Algorithm comparison chart')
        with gr.Tab('🧪 What-if Analysis'):
            gr.Markdown('Explore how hypothetical input changes affect the model output. This is not causal analysis.')
            with gr.Row():
                with gr.Column():
                    ws=gr.Dropdown(['GP','MS'],value='GP',label='School'); wx=gr.Dropdown(['F','M'],value='F',label='Sex')
                    wa=gr.Slider(15,22,value=17,step=1,label='Age'); wd=gr.Dropdown(['U','R'],value='U',label='Home Address')
                    wt=gr.Dropdown(choices=[('Less than 2 hours',1),('2 to 5 hours',2),('5 to 10 hours',3),('More than 10 hours',4)],value=2,label='Weekly Study Time')
                with gr.Column():
                    wf=gr.Slider(0,4,value=0,step=1,label='Past Class Failures'); wss=gr.Dropdown(['yes','no'],value='no',label='School Support')
                    wfs=gr.Dropdown(['yes','no'],value='yes',label='Family Support'); wi=gr.Dropdown(['yes','no'],value='yes',label='Internet Access')
                    wab=gr.Slider(0,75,value=2,step=1,label='Number of Absences')
            wb=gr.Button('Run What-if Analysis',variant='primary'); wt_out=gr.Markdown(); wtable=gr.Dataframe(interactive=False,label='Hypothetical scenarios')
            wb.click(what_if,[ws,wx,wa,wd,wt,wf,wss,wfs,wi,wab],[wt_out,wtable])
        with gr.Tab('📈 Model Performance'): gr.Markdown(performance)
        with gr.Tab('ℹ️ About the Project'):
            gr.Markdown('''## Project Overview
This academic prototype predicts student pass/fail outcomes using Random Forest and provides a separate six-algorithm research comparison.

## Explainable AI
- **SHAP:** Estimates feature contributions to a model prediction.
- **LIME:** Builds a local interpretable approximation around one prediction.

## Dataset
UCI Student Performance — Mathematics dataset.

## Important Limitations
- Predictions are estimates, not guaranteed outcomes.
- What-if results show model sensitivity, not causality.
- The dataset is relatively small and may not represent every student population.
- Do not use this prototype as the sole basis for decisions about students.''')
    gr.Markdown('---\n**Academic prototype | Student Performance Prediction using Explainable AI**')

if __name__=='__main__':
    demo.launch(server_name='0.0.0.0',server_port=int(os.environ.get('PORT',7860)),share=False)
