# -*- coding: utf-8 -*-
import os
import joblib
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "top20_lasso_pipeline.pkl")
FEAT_PATH  = os.path.join(BASE, "top20_features.txt")

st.set_page_config(page_title="Willingness to HLMC", page_icon="🧬", layout="centered")
st.markdown(
    "<h2 style='text-align:center'>Willingness to HLMC</h2>"
    "<p style='text-align:center;color:#666'>Answer each question (Yes / No). Click <b>Next</b> to continue; press <b>Calculate Probability</b> on the last question.</p>",
    unsafe_allow_html=True
)

# ------ Load model & features ------
if not (os.path.exists(MODEL_PATH) and os.path.exists(FEAT_PATH)):
    st.error("Missing model or feature file. Keep app.py, top20_lasso_pipeline.pkl, top20_features.txt in the SAME folder.")
    st.stop()

pipe  = joblib.load(MODEL_PATH)
feats_file = [x.strip() for x in open(FEAT_PATH, "r", encoding="utf-8").read().splitlines() if x.strip()]

# TRUE training order from fitted model (most reliable)
try:
    TRAIN_ORDER = list(pipe.named_steps["clf"].feature_names_in_)
except Exception:
    TRAIN_ORDER = feats_file.copy()

# ------ Your final display order ------
ORDER = [
    "C2_HLMC_heard_1.0",
    "C1_HLM_heard_4.0",
    "B19_SupportHSLS_4.0",
    "B19_SupportHSLS_5.0",
    "B18_SupportHS_5.0",
    "C5_HLMC_whyinterest_Toimprovemyhealthandwellbeing",
    "C5_HLMC_whyinterest_Toaddressaspecificmedicalconcern",
    "C6_HLMC_intervention_Dietaryadvice",
    "C6_HLMC_intervention_Supplements",
    "C6_HLMC_intervention_Mentalhealthinterventions",
    "C8_HLMC_drug_4.0",
    "C7_HLMC_supplements_5.0",
    "C4_HLMC_barrier_Ithinkthecostwouldbetoohigh",
    "C4_HLMC_barrier_Iworrythatthewaitinglistwouldbetoolong",
    "C4_HLMC_barrier_Idon'tseethemasahighpriority",
    "C4_HLMC_barrier_Idon'thaveenoughinformationaboutthem",
    "D5_Health_pay_1.0",
    "C15_Patience_5.0",
    "D9_Aspirations_5_3.0",
]

# Display order = ORDER ∩ TRAIN_ORDER，再加遗漏项（兜底）
feats = [f for f in ORDER if f in TRAIN_ORDER] + [f for f in TRAIN_ORDER if f not in ORDER]
nq = len(feats)

# ------ Questions (Yes/No) ------
QUESTION_MAP = {
    "C2_HLMC_heard_1.0": "Have you heard of HLMC (Healthy Longevity Medicine Clinic)?",
    "C1_HLM_heard_4.0": "Do you believe Healthy Longevity Medicine is very effective?",
    "B19_SupportHSLS_4.0": "Are you likely to support extending both healthspan and lifespan?",
    "B19_SupportHSLS_5.0": "Do you definitely support extending both healthspan and lifespan?",
    "B18_SupportHS_5.0": "Do you support interventions that extend healthspan but not lifespan?",
    "C5_HLMC_whyinterest_Toimprovemyhealthandwellbeing": "Is this a reason for your HLMC interest: to improve your health and wellbeing?",
    "C5_HLMC_whyinterest_Toaddressaspecificmedicalconcern": "Is this a reason for your HLMC interest: to address a specific medical concern?",
    "C6_HLMC_intervention_Dietaryadvice": "Would you want HLMC to provide dietary advice?",
    "C6_HLMC_intervention_Supplements": "Would you want HLMC to provide supplement-based interventions?",
    "C6_HLMC_intervention_Mentalhealthinterventions": "Would you want HLMC to provide mental-health interventions?",
    "C8_HLMC_drug_4.0": "Are you somewhat comfortable with taking prescription medication as an HLMC intervention?",
    "C7_HLMC_supplements_5.0": "Are you extremely comfortable with taking supplements as an HLMC intervention?",
    "C4_HLMC_barrier_Ithinkthecostwouldbetoohigh": "Is this a barrier for you: the cost would be too high?",
    "C4_HLMC_barrier_Iworrythatthewaitinglistwouldbetoolong": "Is this a barrier for you: worrying that the waiting list would be too long?",
    "C4_HLMC_barrier_Idon'tseethemasahighpriority": "Is this a barrier for you: you do not see HLMC as a high priority?",
    "C4_HLMC_barrier_Idon'thaveenoughinformationaboutthem": "Is this a barrier for you: not having enough information about them?",
    "D5_Health_pay_1.0": "Do you pay out-of-pocket for health services?",
    "C15_Patience_5.0": "Are you definitely willing to give up short-term benefits for greater future gains?",
    "D9_Aspirations_5_3.0": "Do you strongly agree with the aspiration to keep yourself healthy and well?",
}

# ------ Session state ------
if "q_idx" not in st.session_state:
    st.session_state.q_idx = 0
if "ans" not in st.session_state:
    st.session_state.ans = {f: None for f in feats}
if "proba" not in st.session_state:
    st.session_state.proba = None

q_idx = st.session_state.q_idx
feature = feats[q_idx]
question = QUESTION_MAP.get(feature, feature)

# Progress
st.progress((q_idx + 1) / nq)
st.markdown(f"<p style='text-align:center;font-weight:600'>Question {q_idx+1} / {nq}</p>", unsafe_allow_html=True)

# Radio (no default)
prev = st.session_state.ans[feature]
options = ["— Select —", "Yes", "No"]
index = 0 if prev is None else (1 if prev == 1.0 else 2)
choice = st.radio(question, options, index=index, horizontal=True)
selected = None if choice == "— Select —" else (1.0 if choice == "Yes" else 0.0)
st.session_state.ans[feature] = selected

# Nav
c1, c2, c3 = st.columns([1,1,1])
with c1:
    if q_idx > 0 and st.button("⬅️ Previous"):
        st.session_state.q_idx -= 1
        st.rerun()
with c2:
    if q_idx < nq - 1:
        if st.button("Next ➡️"):
            if st.session_state.ans[feature] is None:
                st.warning("Please select Yes or No before continuing.")
            else:
                st.session_state.q_idx += 1
                st.rerun()
    else:
        if st.button("Calculate Probability ✅"):
            if any(st.session_state.ans[f] is None for f in feats):
                st.warning("Please answer all questions first.")
            else:
                # Build inputs from display order
                row = {f: float(st.session_state.ans[f]) for f in feats}
                X = pd.DataFrame([row])

                # Self-check & reorder to training order
                set_train, set_now = set(TRAIN_ORDER), set(X.columns)
                missing = [c for c in TRAIN_ORDER if c not in set_now]
                extra   = [c for c in X.columns if c not in set_train]
                if missing or extra:
                    st.error("Feature mismatch with the trained model.")
                    if missing: st.write("Missing:", missing)
                    if extra:   st.write("Unexpected:", extra)
                    st.stop()

                X = X.reindex(columns=TRAIN_ORDER).astype(float)
                st.session_state.proba = float(pipe.predict_proba(X)[0, 1])
                st.balloons()
                st.rerun()
with c3:
    if st.button("🔄 Restart"):
        st.session_state.q_idx = 0
        st.session_state.ans  = {f: None for f in feats}
        st.session_state.proba = None
        st.rerun()

# Result
if st.session_state.proba is not None:
    st.success(f"Predicted probability of attending the clinic: **{st.session_state.proba:.3f}**")
    st.progress(int(st.session_state.proba * 100))
