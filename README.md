# 🏦 CreditRisk AI - Loan Underwriting Copilot

**Live app:** [paste Streamlit link here]
**Course:** Data Science for Managers using Python - IIM Bodh Gaya | Team [name]

## The business problem
Loan officers at banks/NBFCs must decide quickly whether to approve a loan.
Approving a future defaulter loses ~60% of the loan; rejecting a good customer loses interest profit.
CreditRisk AI gives a risk score, an APPROVE / REVIEW / REJECT decision and the reasons behind it,
in seconds.

## What the app does
- Probability of default for any applicant (tuned XGBoost)
- Decision bands chosen to **minimise business cost**, not a default 0.5 cut-off
- Top reasons for each decision (SHAP) - an audit trail for compliance
- What-if simulator (change loan amount, income, rate)
- AI-written decision letter (Gemini), with a template fallback
- Portfolio ROI dashboard

## Results (unseen 20% test set)
| Model | CV ROC-AUC (5-fold) |
|---|---|
| Logistic Regression | [x.xxx ± x.xxx] |
| Random Forest | [x.xxx ± x.xxx] |
| **XGBoost (tuned)** | **[x.xxx ± x.xxx]** |

Test ROC-AUC: **[x.xxx]** · Saving vs approving everyone: **$[xx,xxx]** on [n] test loans

## Architecture
Data (Kaggle Credit Risk Dataset, 32.5k loans)
→ `src/data.py`: clean, stratified 80/20 split, impute + scale + one-hot (inside a Pipeline, no leakage)
→ `src/model.py`: 5-fold CV of 3 models → RandomizedSearchCV tuning → cost-based threshold → SHAP
→ `models/model.joblib`
→ `app.py`: Streamlit UI (prediction, SHAP reasons, what-if, Gemini letter, ROI)

## Run it yourself
```bash
git clone [your repo URL]
cd creditrisk-ai
pip install -r requirements.txt

# Re-train (optional - a trained model is already included)
python -m src.model cv
python -m src.model tune
python -m src.model roi
python -m src.model shap

# Launch the app
streamlit run app.py
```
Optional: for the AI letter, create `.streamlit/secrets.toml` with `GEMINI_API_KEY = "..."`.

## Key assumptions
Loss given default 60%, cost of funds 7%, one-year loans. Amounts are in the dataset's currency (USD).

## Limitations
- US-style dataset; would need re-training on Indian bureau data (e.g. CIBIL)
- `loan_grade` and `loan_int_rate` are set by the lender, so they partly encode an earlier risk judgement
- Needs fairness checks (e.g. age) and periodic re-training as economic conditions change

## Team
[Names]

## How we built it
See `VIBE_CODING_LOG.md` for the AI tools and prompts we used.