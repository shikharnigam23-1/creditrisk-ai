# Vibe Coding Log — CreditRisk AI

*Course: Data Science for Managers using Python, IIM Bodh Gaya*
*Repository: `creditrisk-ai` (GitHub remote `origin`, branch `main`)*
*Evidence used: project files, the generated `reports/` folder, and the Git history (13 commits on 7 October 2026). Anything that could not be checked against these is marked "Not available in the supplied project history."*

---

## 1. Project Overview

CreditRisk AI is a loan-underwriting assistant for a credit officer at a bank or NBFC. The officer enters an applicant's details and the app returns:

- the **probability of default**,
- a decision: **APPROVE**, **REVIEW** or **REJECT**,
- the **top reasons** behind the decision (from SHAP),
- a **what-if simulator**,
- an **AI-drafted decision letter**.

**Why we built it:** a lender can make two expensive mistakes. Approving a borrower who later defaults loses a large part of the loan. Rejecting a borrower who would have repaid loses the interest margin. The project therefore tries to (a) predict default risk accurately and (b) set the approval cut-offs where the total cost of both mistakes is lowest, instead of using the default 0.5 cut-off.

**Data:** the public Kaggle *Credit Risk Dataset* (`data/credit_risk_dataset.csv`, 32,581 loan records plus a header row). The target is `loan_status`, where 1 = default and 0 = repaid.

## 2. Development Approach

The project was built in small, numbered tasks (T1–T11b). Each task ended with a working result and its own Git commit. All 13 commits are by Shikhar Nigam on 7 October 2026, between 14:38 and 17:13 IST.

| Commit | Time (IST) | Message | Files changed |
| --- | --- | --- | --- |
| `45b5bf0` | 14:38 | T1: Setup credit risk AI project | `.gitignore` |
| `b213966` | 14:54 | T2: Add credit risk dataset | `data/credit_risk_dataset.csv` |
| `1a671cf` | 14:58 | T3: Add data loading and preprocessing pipeline | `src/__init__.py`, `src/data.py` |
| `9fb0730` | 15:07 | Fix NaN filter bug, add get_data and build_preprocessor | `src/data.py` |
| `70765cf` | 15:40 | T4: 5-fold CV comparison of 3 models | `src/model.py`, `reports/cv_results.csv`, `reports/cv_comparison.png` |
| `14befcb` | 16:12 | T5: tuned XGBoost + test evaluation | `src/model.py`, `models/model.joblib`, 4 report files |
| `702de31` | 16:16 | T6: cost-based threshold + ROI | `src/model.py`, 3 report files |
| `0f7fa75` | 16:18 | T7: SHAP explainability + explain() for app | `src/model.py`, 3 SHAP charts |
| `eaf016f` | 16:29 | T8: Streamlit app v1 - form, risk score, decision | `app.py` |
| `c866eb7` | 16:32 | T9: SHAP reasons + what-if simulator | `app.py` |
| `ef5c5e4` | 16:55 | T10: Gemini decision letter with fallback + ROI dashboard | `app.py`, `.gitignore` |
| `f0a99d4` | 17:08 | T11a: pinned requirements + gitignore | `requirements.txt`, `.gitignore` |
| `dfeb165` | 17:13 | T11b: README | `README.md` |

**Working pattern for each task:**

1. Plan the task and get a code draft with an AI assistant.
2. Paste it into the project in VS Code.
3. Run it from the terminal.
4. Check the printed output and the files it saved.
5. Commit.

The code was kept modular so each piece could be run and checked on its own:

- `src/data.py` handles data loading and preparation.
- `src/model.py` handles training, evaluation, thresholds and explainability. It runs one stage at a time with `python -m src.model cv | tune | roi | shap`.
- `app.py` is the Streamlit interface.

Outputs are saved to `models/` and `reports/`, so the app and the slides read the same numbers.

> **Note on scope:** T1 and T2 (setup and dataset) and T3 (`data.py`) came before the tasks this log covers in detail. They are summarised here because later tasks depend on them.

---

## 3. T4 — Model Comparison

### Goal
Compare three classifiers fairly before choosing one, rather than assuming the most popular model is best.

### AI-assisted development
Claude (used through a chat session) proposed the structure of `src/model.py`. This included the `make_pipeline()` helper that puts preprocessing and the model into one scikit-learn `Pipeline`, plus the cross-validation loop. The version committed in `70765cf` follows that structure but is reformatted, with one argument per line and extra step-by-step comments.

Which tool or person did the reformatting is Not available in the supplied project history.

### Implementation
- **Three candidate models:**
  - Logistic Regression with `class_weight="balanced"`
  - Random Forest with 300 trees, balanced
  - XGBoost with 300 trees, depth 5, and `scale_pos_weight` set to the good-to-default ratio in the training data
- **Shared setup:** all three use the same preprocessing pipeline from `data.py`, which applies median imputation and scaling to numeric columns and one-hot encoding to categorical columns.
- **Evaluation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the training data only, scored on ROC-AUC, precision, recall and F1.

```python
def make_pipeline(model):
    # Preprocessing + model in ONE object, so each CV fold fits
    # imputation/scaling only on its own training part (no leakage).
    return Pipeline([("prep", build_preprocessor()), ("model", model)])
```

### Validation
- **Run locally:** the developer ran `python -m src.model cv`, and the results were committed with the code (`70765cf`).
- **Stability:** the standard deviation across folds is the stability check. XGBoost's ROC-AUC varied by only ±0.0046 across the five folds.

### Output
`reports/cv_results.csv` (5-fold CV on the training set):

| Model | ROC-AUC (mean ± std) | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| XGBoost | 0.9481 ± 0.0046 | 0.8191 | 0.8014 | 0.8101 |
| Random Forest | 0.9317 ± 0.0051 | 0.9742 | 0.7151 | 0.8247 |
| Logistic Regression | 0.8722 ± 0.0049 | 0.5506 | 0.7766 | 0.6442 |

The chart is in `reports/cv_comparison.png`. XGBoost had the highest ROC-AUC, so it went forward to tuning. Random Forest had a slightly higher F1 at the default 0.5 cut-off, but ROC-AUC was the agreed selection metric because the final cut-off is chosen separately in T6.

---

## 4. T5 — XGBoost Tuning

### Goal
Improve XGBoost by searching its settings, then measure it once on data it has never seen.

### AI-assisted development
Claude drafted `run_tune()`, including the search space with a one-line comment on what each hyperparameter controls. The developer pasted it into `src/model.py` (commit `14befcb`, +55 lines).

### Implementation
- **Search:** `RandomizedSearchCV` with `n_iter=25` over the same 5 folds (125 model fits), optimising ROC-AUC.
- **Search space:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, and `scale_pos_weight` (either 1.0 or the class ratio).
- **Test evaluation:** the best pipeline was scored once on the 20% test set, which had been held out since `get_data()` split it with `random_state=42`.
- **Error analysis:** compares the average profile of defaulters the model missed with defaulters it caught.

### Validation
- **Overfitting check:** the best CV ROC-AUC was 0.949 and the test ROC-AUC was 0.9455. The small gap suggests the model generalises.
- **Recall at 0.5 was only about 72%.** That is the reason the next task (T6) replaces the 0.5 cut-off with a cost-based one.

### Output
`reports/test_metrics.json`:

| Item | Value |
| --- | --- |
| Best CV ROC-AUC | 0.949 |
| Best parameters | `n_estimators=700`, `max_depth=4`, `learning_rate=0.05`, `subsample=1.0`, `colsample_bytree=0.6`, `min_child_weight=5`, `scale_pos_weight=1.0` |
| Test ROC-AUC | 0.9455 |
| Test precision / recall / F1 (at 0.5) | 0.964 / 0.7171 / 0.8224 |

`reports/error_analysis.csv` shows that **missed defaulters look safer on paper** than caught ones:

| Average | Missed defaulters | Caught defaulters |
| --- | --- | --- |
| Income | $60,170 | $45,964 |
| Loan amount | $9,084 | $11,810 |
| Loan as % of income | 16% | 28% |
| Interest rate | 11.77% | 13.52% |

Other files saved: `models/model.joblib` (preprocessing + tuned model as one object), `reports/roc_curve.png` and `reports/confusion_matrix.png`.

---

## 5. T6 — Cost-Based Decision Threshold

### Goal
Turn probabilities into lending decisions that minimise money lost, not classification errors.

### Business reasoning
The two mistakes do not cost the same:

| Mistake | Cost assumed in `src/model.py` |
| --- | --- |
| Approving a borrower who defaults | loan × 60% (`LGD = 0.60`, loss given default) |
| Rejecting a borrower who would repay | loan × (interest rate − 7%) (`COST_OF_FUNDS = 0.07`), with a floor of 1% |

Missing a defaulter is far more expensive than turning away a good customer. A lender should therefore approve only clearly low-risk applicants and send the uncertain middle to a person.

**These rates are planning assumptions written into the code, not estimates from data.** They are stated in `roi.json` so they can be challenged.

### Implementation
- **Threshold:** chosen on out-of-fold training predictions (`cross_val_predict`), never on the test set, by searching cut-offs from 0.05 to 0.95 in steps of 0.01.
- **Decision bands:** placed around the best cut-off at ±0.10, with a minimum of 0.02:
  - **APPROVE** below the lower edge
  - **REJECT** at or above the upper edge
  - **REVIEW** in between
- **Benchmarks on the test set:** compared against "approve everyone" and a simple human-style rule, "reject loan grades D–G".

### Validation
- **Business sense:** the saved band table shows the expected ordering, with almost no defaults among approvals and most defaults among rejections (see the table below).
- **Comparison against both benchmarks:** the model's cost is reported against both simple policies, not just one.

### Output
`reports/roi.json` and `reports/decision_bands.csv` (6,515 test loans):

| Item | Value |
| --- | --- |
| Cost-optimal cut-off | 0.08 |
| Bands | APPROVE < 2%, REVIEW 2–18%, REJECT ≥ 18% |
| Cost if approving everyone | $9,412,005 |
| Cost with "reject grades D–G" rule | $5,563,904 |
| Cost with CreditRisk AI | $1,244,719 |
| Saving vs approving everyone | $8,167,286 |
| Saving vs grade rule | $4,319,184 |
| Saving per 1,000 applications | $1,253,613 |

| Band | Applicants | Share | Actual default rate |
| --- | --- | --- | --- |
| APPROVE | 1,598 | 24.5% | 0.6% |
| REVIEW | 3,229 | 49.6% | 7.0% |
| REJECT | 1,688 | 25.9% | 70.2% |

The cost-by-threshold chart is saved as `reports/threshold_cost_curve.png`. Amounts are in the dataset's units (USD), as `roi.json` notes.

---

## 6. T7 — SHAP Explainability

### Goal
Explain both what drives risk in general and why one specific applicant got their decision. A lender has to be able to justify a rejection.

### Implementation
- **`run_shap()`:** uses SHAP's `TreeExplainer` on a sample of 1,000 test loans and saves three charts: a beeswarm, a bar chart of average impact, and a waterfall for the riskiest applicant in the sample.
- **`explain(applicant)`:** the function the app uses.
  - It reads XGBoost's built-in SHAP contributions (`pred_contribs=True`).
  - It adds the one-hot columns back together under their original field. For example, `person_home_ownership_RENT` and `person_home_ownership_OWN` become *home ownership*.
  - It returns the top reasons, each marked "raises risk" or "lowers risk".

Using XGBoost's built-in contributions in the app means the live app does not depend on the SHAP plotting library.

### Validation
- **Run locally:** `python -m src.model shap` was run and its three charts committed (`0f7fa75`).
- **Consistency with T5:** the global ranking puts income and loan-to-income at the top. This matches the T5 error analysis, where those were the two biggest differences between missed and caught defaulters.
- **Recorded review of individual explanations:** not available in the supplied project history.

### Output
`reports/shap_summary.png`, `reports/shap_importance.png` and `reports/shap_example_applicant.png`.

Top features by mean |SHAP value| in `shap_importance.png`, shown per encoded column:

| Feature | Mean impact |
| --- | --- |
| Person income | 0.91 |
| Loan as % of income | 0.77 |
| Interest rate | 0.45 |
| Home ownership = RENT | 0.38 |
| Home ownership = OWN | 0.38 |
| Loan grade = D | 0.33 |

---

## 7. T8 — Streamlit Application

### Goal
Put the model in front of a non-technical credit officer.

### Implementation
`app.py` (commit `eaf016f`, 173 lines) contains:

- **Model loading:** `@st.cache_resource` loads `model.joblib` once, so the app stays fast.
- **Decision cut-offs:** read from `reports/roi.json`, so they are not hard-coded.
- **Sidebar form:** 10 inputs. `loan_percent_income` is calculated from loan ÷ income rather than typed in.
- **Demo presets:** four preset applicants (*Low risk – salaried homeowner*, *Borderline – young renter*, *High risk – stretched borrower*, *Custom applicant*), so the demo can be repeated reliably.
- **Main panel:** probability of default, the colour-coded decision, loan-to-income ratio, and a line stating the policy.
- **Placeholders:** marked `# ===== T9 SECTION =====` and `# ===== T10 SECTION =====`, so the app ran end to end before the later features existed.

### Testing
- **How to test:** the app is designed to be checked by running `streamlit run app.py` and selecting each preset.
- **Recorded test results:** Not available in the supplied project history.

---

## 8. T9 — What-If Simulator and SHAP Reasons

### Goal
Turn a single score into an explanation and something the officer can act on.

### Implementation
In commit `c866eb7` (+48 / −4 lines), the T9 placeholder section was replaced with:

- **`show_reasons()`:** shows the top three reasons from `explain()` as text with up/down markers, plus a red/green bar chart of the top five.
- **`show_what_if()`:** sliders for loan amount, income and interest rate. They recalculate loan-to-income, re-score the applicant, and show the change in risk and the new decision.

The diff shows only the placeholder lines were removed, which confirms the layered build approach.

### Testing
- **How to test:** move each slider on a preset applicant and watch the risk and decision update.
- **Recorded test results**, including which changes flipped a decision: Not available in the supplied project history.

---

## 9. T10 — Gemini Decision Letter and ROI Dashboard

### Goal
- Use generative AI in a controlled way: it writes a letter, it does not decide.
- Show the business value inside the product.

### Implementation
- **`show_letter()`:**
  - Builds a prompt from the decision and the top three SHAP reasons.
  - Tells the model not to mention models, SHAP or probabilities.
  - Calls Gemini through the `google-genai` library (`GEMINI_MODEL = "gemini-2.5-flash"`).
  - Shows the letter in an editable text box with a note that a human reviews it before sending.
- **`show_roi()`:**
  - Reads `roi.json` and shows cost with and without the model plus the saving against the grade rule.
  - Includes a portfolio-size input that scales the saving.
  - Shows the decision-band table and the saved evidence charts.
- **API key:** read with `st.secrets["GEMINI_API_KEY"]`, falling back to an environment variable. The key itself is not in the repository.

### Fallback design
The demo must not break if the AI service is down or no key is set:

```python
if key:
    try:
        ...  # call Gemini
    except Exception as e:
        st.warning(f"AI service unavailable ({type(e).__name__}), so the standard template was used.")
if not letter:
    letter = _template_letter(decision, reasons)
```

`_template_letter()` writes a standard APPROVE, REVIEW or REJECT letter that uses the same top factors.

### Testing
- **Secret protection:** commit `ef5c5e4` also added `.streamlit/secrets.toml` to `.gitignore`.
- **Letter generation and fallback:** whether a Gemini letter was generated successfully, and whether the fallback was tested, is Not available in the supplied project history.

---

## 10. T11a — Packaging

**`requirements.txt`** (commit `f0a99d4`) pins exact versions:

| Package | Version |
| --- | --- |
| streamlit | 1.50.0 |
| pandas | 2.3.3 |
| numpy | 2.0.2 |
| scikit-learn | 1.6.1 |
| xgboost | 2.1.4 |
| shap | 0.49.1 |
| joblib | 1.5.3 |
| matplotlib | 3.9.4 |
| google-genai | 1.47.0 |

**Why pinning matters:** `model.joblib` stores a fitted scikit-learn and XGBoost pipeline. It may fail to load under different versions of those two libraries, so the deployment environment has to match the training environment.

**`.gitignore`** was rewritten in the same commit into three labelled groups:

- **Secrets:** `.streamlit/secrets.toml`, `.env`
- **Python clutter:** `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `.ipynb_checkpoints/`
- **Mac clutter:** `.DS_Store`

`models/`, `reports/` and `data/` were deliberately **not** ignored, because the deployed app needs the model and report files.

**Secret protection:**
- The Git history contains no `secrets.toml`, `.env` or `.streamlit/` file in any commit.
- `.DS_Store` exists in the working folder but is not tracked.

---

## 11. T11b — Documentation

Commit `dfeb165` ("T11b: README") added `README.md`, but **the committed file is empty (0 bytes)**, and the working copy is empty too.

The README's intended content is therefore not available in the supplied project history. Writing it (problem, results from `reports/`, architecture, run commands, assumptions and limitations, live app link) is an open task before submission.

---

## 12. AI Tools and Techniques Used

| Tool | Evidence | Why it was used | What it helped with | How the developer validated the output |
| --- | --- | --- | --- | --- |
| **Claude** (chat) | The task numbering (T1–T11b), the `python -m src.model cv/tune/roi/shap` structure, the `T9 SECTION` / `T10 SECTION` placeholders and the commit messages match the step-by-step plan produced in the Claude session | Planning a one-night build and drafting code for each task | Task breakdown; drafts of `model.py` (T4–T7) and `app.py` (T8–T10); finding the NaN-filter bug in `data.py`; `requirements.txt` and `.gitignore` guidance | Ran each stage locally and committed its outputs together with the code; review steps beyond that are Not available in the supplied project history |
| **Google Gemini** (in the product, not a coding tool) | `app.py`: `google-genai` client, `GEMINI_MODEL = "gemini-2.5-flash"` | Drafting the plain-English decision letter | Turning the decision and SHAP reasons into a letter | Prompt limits it to the given facts; template fallback; officer reviews every letter |
| GitHub Copilot, Google Antigravity, ChatGPT/Codex | Not available in the supplied project history | — | — | — |

**Techniques used:**

- **Task-sized prompting:** one task per request, each ending in a runnable result and a commit.
- **Placeholder-first building:** the app ran with stub sections, which were then replaced one at a time.
- **Code review by AI:** the AI reviewed the developer's own `data.py` and found the NaN bug.

The original `src/data.py` (T3) and the reformatted T4 code are written in a different style from the AI drafts. Who or what wrote them is Not available in the supplied project history.

---

## 13. Human-in-the-Loop Development

| Area | What the developer did, not the AI |
| --- | --- |
| Ownership of data prep | Wrote or supplied the T3 `data.py` and shared it for review, then applied and committed the fix (`9fb0730`) |
| Running the pipeline | Ran every stage on their own machine; every report in `reports/` was generated locally and committed with the code that produced it |
| Model choice | Took XGBoost forward on the basis of their own CV results; ROC-AUC was the selection metric |
| Overfitting check | Compared CV ROC-AUC (0.949) with test ROC-AUC (0.9455) |
| Business assumptions | Kept the 60% loss-given-default and 7% cost-of-funds assumptions explicit and visible in `roi.json` |
| Version control | Made 13 small commits with task-level messages; cleaned up `.gitignore` duplicates |
| Secret handling | Kept the Gemini key out of Git; no secret file appears anywhere in the history |
| AI limits | Designed the app so Gemini only drafts text, with a template fallback and human review |

---

## 14. Key Debugging / Problem-Solving

Only issues backed by the code or Git history are listed.

**1. Rows with missing employment length were being silently dropped (commit `9fb0730`)**

- **Symptom found in review:** the T3 filter `df[df["person_emp_length"] <= 60]` also deleted every row where employment length was blank. In pandas, `NaN <= 60` evaluates to False.
- **Fix:** keep missing values and let the median imputer fill them:
  ```python
  df = df[df["person_emp_length"].isna() | (df["person_emp_length"] <= 60)].copy()
  ```
- **Same commit:**
  - Changed numeric column detection from `["int64", "float64"]` to `"number"`, so no numeric type is missed.
  - Moved the split settings into constants (`TEST_SIZE`, `RANDOM_STATE`).
  - Added `get_data()` and `build_preprocessor()`, so `model.py` and `app.py` reuse exactly the same split and preprocessing.

**2. Duplicate entry in `.gitignore` (commits `ef5c5e4` → `f0a99d4`)**

- **Problem:** T10 appended `.streamlit/secrets.toml` even though T1 had already added it, leaving the line twice and without a final newline.
- **Fix:** T11a rewrote `.gitignore` into clean, commented groups.

**3. Empty README (commit `dfeb165`)**

- `README.md` was committed with no content. This is unresolved in the supplied history.

**4. Deployment**

- Errors or fixes during deployment to Streamlit Community Cloud (for example, Python version or build time) are not available in the supplied project history.

---

## 15. Lessons Learned

| Topic | Lesson |
| --- | --- |
| AI-assisted coding | AI drafts are fast, but they must be run, read and checked. Small tasks with one commit each made it easy to see what changed and to roll back. |
| ML pipelines | Putting preprocessing and the model in one `Pipeline` and saving it with `joblib` means the app transforms inputs exactly as training did. |
| Data leakage | Fit imputation and scaling inside each CV fold, choose the threshold on training predictions, and touch the test set once. |
| Data cleaning | A filter on a column with missing values can silently delete rows. Always print row counts before and after cleaning. |
| Model evaluation | With about 22% defaults, accuracy alone misleads. ROC-AUC compares models; recall shows how many defaulters are caught. CV spread shows stability. |
| Explainability | Grouping one-hot columns back into their original fields makes SHAP reasons readable for a manager. |
| Business-cost modelling | The best cut-off here was 0.08, not 0.5. The model's value appeared only when the decision rule was tied to money. |
| Streamlit development | Caching the model, reading cut-offs from a file, and building with placeholders kept the app fast and always runnable. |
| API key security | Keep keys in `secrets.toml` or the hosting platform's Secrets setting, git-ignore them from the first commit, and design a fallback for when the API fails. |

---

## 16. Final Project

The final system is a modular Python repository:

```
creditrisk-ai/
├── app.py                    # Streamlit app: form, decision, SHAP reasons, what-if, letter, ROI tab
├── src/data.py               # load, clean, stratified 80/20 split, preprocessing
├── src/model.py              # cv, tune, roi, shap stages + explain() used by the app
├── models/model.joblib       # tuned XGBoost pipeline (preprocessing + model)
├── data/credit_risk_dataset.csv
├── reports/                  # CV table, test metrics, ROI, bands, error analysis, 7 charts
├── requirements.txt          # pinned versions
├── .gitignore                # secrets excluded
└── README.md                 # committed empty; to be completed
```

**Final model:** tuned XGBoost with a test ROC-AUC of 0.9455. Decisions use cost-based bands (approve below 2%, reject at 18% or above). On the 6,515 test loans this lowered the modelled cost of wrong decisions from $9.41M (approve everyone) to $1.24M.

Whether the app is live on Streamlit Community Cloud, and at what link, is Not available in the supplied project history.
