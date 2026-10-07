"""
app.py - CreditRisk AI: loan underwriting copilot (Streamlit)
Run from the project root:   streamlit run app.py
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.model import MODEL_PATH, REPORTS, explain

st.set_page_config(page_title="CreditRisk AI", page_icon="🏦", layout="wide")


# ---------- Load the trained model and the decision bands ----------
@st.cache_resource          # load the model once, keep it in memory (fast app)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_roi():
    path = REPORTS / "roi.json"
    return json.loads(path.read_text()) if path.exists() else None


model = load_model()
roi = load_roi()
# Cost-optimised cut-offs chosen in T6 (fallback values if roi.json is missing)
APPROVE_BELOW = roi["bands"]["approve_below"] if roi else 0.30
REJECT_ABOVE = roi["bands"]["reject_above"] if roi else 0.50


# ---------- Labels and demo applicants ----------
FRIENDLY = {
    "person_age": "Age",
    "person_income": "Annual income",
    "person_home_ownership": "Home ownership",
    "person_emp_length": "Years employed",
    "loan_intent": "Loan purpose",
    "loan_grade": "Loan grade",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan as % of income",
    "cb_person_default_on_file": "Previous default on file",
    "cb_person_cred_hist_length": "Credit history length (yrs)",
}
HOME = ["RENT", "OWN", "MORTGAGE", "OTHER"]
INTENTS = ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
GRADES = ["A", "B", "C", "D", "E", "F", "G"]

PRESETS = {
    "✅ Low risk - salaried homeowner": dict(age=38, income=95000, home="MORTGAGE", emp=10,
        intent="HOMEIMPROVEMENT", grade="A", amount=8000, rate=7.5, prior="N", hist=12),
    "🟠 Borderline - young renter": dict(age=26, income=42000, home="RENT", emp=3,
        intent="PERSONAL", grade="C", amount=10000, rate=13.5, prior="N", hist=4),
    "⛔ High risk - stretched borrower": dict(age=23, income=24000, home="RENT", emp=1,
        intent="MEDICAL", grade="D", amount=12000, rate=16.0, prior="Y", hist=2),
    "✏️ Custom applicant": dict(age=30, income=50000, home="RENT", emp=5,
        intent="EDUCATION", grade="B", amount=10000, rate=11.0, prior="N", hist=5),
}


# ---------- Core helper functions ----------
def make_applicant(age, income, home, emp, intent, grade, amount, rate, prior, hist):
    """Turn form inputs into a 1-row DataFrame with the SAME columns used in training."""
    return pd.DataFrame([{
        "person_age": age, "person_income": income, "person_home_ownership": home,
        "person_emp_length": emp, "loan_intent": intent, "loan_grade": grade,
        "loan_amnt": amount, "loan_int_rate": rate,
        "loan_percent_income": round(amount / max(income, 1), 2),
        "cb_person_default_on_file": prior, "cb_person_cred_hist_length": hist,
    }])


def predict(applicant):
    """Probability of default (0 to 1) from the tuned XGBoost pipeline."""
    return float(model.predict_proba(applicant)[:, 1][0])


def decide(p):
    """Turn a probability into a business decision using the cost-optimised bands."""
    if p < APPROVE_BELOW:
        return "APPROVE", "green"
    if p >= REJECT_ABOVE:
        return "REJECT", "red"
    return "REVIEW", "orange"


def fmt(feature, value):
    """Show values in business-friendly format."""
    if feature in ("person_income", "loan_amnt"):
        return f"${value:,.0f}"
    if feature == "loan_percent_income":
        return f"{value:.0%}"
    if feature == "loan_int_rate":
        return f"{value:.1f}%"
    if feature == "cb_person_default_on_file":
        return "Yes" if value == "Y" else "No"
    return str(value).title()


# ===== T9 SECTION =====
def show_reasons(applicant):
    """Top drivers of THIS applicant's risk, using SHAP values from explain() in model.py."""
    st.subheader("Why this decision? (top drivers, from SHAP)")
    reasons = explain(applicant, model, top_n=5)

    for r in reasons[:3]:
        name = FRIENDLY.get(r["feature"], r["feature"])
        icon = "🔺" if r["impact"] > 0 else "🟢"
        st.markdown(f"{icon} **{name}** = `{fmt(r['feature'], r['value'])}` → {r['effect']}")

    labels = [FRIENDLY.get(r["feature"], r["feature"]) for r in reasons][::-1]
    impacts = [r["impact"] for r in reasons][::-1]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh(labels, impacts, color=["#dc2626" if v > 0 else "#16a34a" for v in impacts])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Impact on default risk   ← lowers | raises →")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("SHAP splits the prediction fairly among the features (Shapley values from game theory). "
               "Red pushes risk up, green pushes it down.")
    return reasons


def show_what_if(applicant, proba):
    """Change loan terms and see risk + decision update live."""
    st.subheader("🎛️ What-if simulator")
    st.caption("Try a smaller loan, a higher income or a different rate: how does the decision change?")
    base = applicant.iloc[0]
    tag = f"{base.loan_amnt}_{base.person_income}_{base.loan_int_rate}"  # resets sliders for a new applicant

    c1, c2, c3 = st.columns(3)
    new_amount = c1.slider("Loan amount ($)", 500, int(max(base.loan_amnt * 2, 1000)),
                           int(base.loan_amnt), step=500, key=f"wi_amt_{tag}")
    new_income = c2.slider("Annual income ($)", 5000, int(max(base.person_income * 2, 10000)),
                           int(base.person_income), step=1000, key=f"wi_inc_{tag}")
    new_rate = c3.slider("Interest rate (%)", 5.0, 25.0, float(base.loan_int_rate),
                         step=0.5, key=f"wi_rate_{tag}")

    what_if = applicant.copy()
    what_if["loan_amnt"] = new_amount
    what_if["person_income"] = new_income
    what_if["loan_int_rate"] = new_rate
    what_if["loan_percent_income"] = round(new_amount / new_income, 2)

    new_p = predict(what_if)
    new_decision, new_colour = decide(new_p)
    m1, m2 = st.columns(2)
    m1.metric("New probability of default", f"{new_p:.1%}",
              delta=f"{(new_p - proba) * 100:+.1f} pts", delta_color="inverse")
    m2.markdown(f"### New decision: :{new_colour}[**{new_decision}**]")
# ===== END T9 SECTION =====


# ===== T10 SECTION (replace this whole section in T10) =====
def show_letter(applicant, proba, decision, reasons):
    st.info("AI decision letter coming in T10.")


def show_roi():
    st.info("Portfolio ROI dashboard coming in T10.")
# ===== END T10 SECTION =====


# ---------- Sidebar: applicant input form ----------
st.sidebar.title("🏦 Applicant details")
preset = st.sidebar.selectbox("Load a demo applicant", list(PRESETS))
d = PRESETS[preset]
k = list(PRESETS).index(preset)  # changing the preset reloads the default values

age = st.sidebar.number_input("Age", 18, 100, d["age"], key=f"age{k}")
income = st.sidebar.number_input("Annual income ($)", 1000, 2_000_000, d["income"], step=1000, key=f"inc{k}")
home = st.sidebar.selectbox("Home ownership", HOME, index=HOME.index(d["home"]),
                            format_func=str.title, key=f"home{k}")
emp = st.sidebar.number_input("Years employed", 0, 60, d["emp"], key=f"emp{k}")
intent = st.sidebar.selectbox("Loan purpose", INTENTS, index=INTENTS.index(d["intent"]),
                              format_func=str.title, key=f"int{k}")
grade = st.sidebar.selectbox("Loan grade (A = safest)", GRADES, index=GRADES.index(d["grade"]), key=f"gr{k}")
amount = st.sidebar.number_input("Loan amount ($)", 500, 500_000, d["amount"], step=500, key=f"amt{k}")
rate = st.sidebar.number_input("Interest rate (%)", 5.0, 25.0, d["rate"], step=0.1, key=f"rate{k}")
prior = st.sidebar.radio("Previous default on file?", ["N", "Y"], index=["N", "Y"].index(d["prior"]),
                         format_func=lambda v: "Yes" if v == "Y" else "No", horizontal=True, key=f"pr{k}")
hist = st.sidebar.number_input("Credit history (years)", 0, 40, d["hist"], key=f"hist{k}")

applicant = make_applicant(age, income, home, emp, intent, grade, amount, rate, prior, hist)
proba = predict(applicant)
decision, colour = decide(proba)


# ---------- Main page ----------
st.title("CreditRisk AI")
st.caption("Loan underwriting copilot: risk score, decision and reasons in seconds.")
tab_assess, tab_roi = st.tabs(["🔍 Assess applicant", "📈 Portfolio ROI"])

with tab_assess:
    c1, c2, c3 = st.columns(3)
    c1.metric("Probability of default", f"{proba:.1%}")
    c2.markdown(f"### Decision: :{colour}[**{decision}**]")
    c3.metric("Loan as % of income", f"{applicant['loan_percent_income'][0]:.0%}")
    st.progress(min(proba, 1.0))
    st.caption(f"Policy: approve below {APPROVE_BELOW:.0%} risk · human review in between · "
               f"reject at {REJECT_ABOVE:.0%}+ (cost-optimised cut-offs from model training).")
    st.divider()
    reasons = show_reasons(applicant)
    st.divider()
    show_what_if(applicant, proba)
    st.divider()
    show_letter(applicant, proba, decision, reasons)

with tab_roi:
    show_roi()