# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small Flask app that predicts credit-card customer churn (attrition) for a bank, using a pre-trained scikit-learn pipeline (`model.pkl`). Single client lookups go through a form; bulk lookups sample random rows from a local SQLite database.

## Commands

### Environment setup

The repo ships a broken `env/` (a venv built against a Python 3.8 Anaconda install that no longer resolves on this machine — ignored by git). Use a fresh venv instead:

```bash
python -m venv env311
env311/Scripts/activate        # Windows
pip install -r requirements.txt
```

`requirements.txt` is now UTF-8 and installs cleanly (Flask, scikit-learn==1.9.0, numpy 2.x, pandas, xgboost/lightgbm/catboost, shap — see "Model training" below for why the boosting libraries are needed even just to *load* `model.pkl`).

### Run the app

```bash
python wsgi.py          # dev server on http://127.0.0.1:5000, debug=False
```

`Procfile` defines the production command (`gunicorn wsgi:app --preload --workers 3`), but gunicorn doesn't run natively on Windows — use WSL/Linux to test that path.

### No test suite

There are no automated tests in this repository.

## Architecture

### Request flow (`app.py`)

Single-file Flask app, six routes:
- `GET /` — home page.
- `GET /Predic_form` + `POST /predict` — single-client prediction. The form's field order **must** match the model's expected feature order: `Gender, Customer_Age, Total_Relationship_Count, Months_Inactive_12_mon, Total_Revolving_Bal, Total_Trans_Amt, Avg_Utilization_Ratio`. `/predict` builds the feature vector from `request.form.values()` positionally (no field-name mapping), so reordering fields in the template silently breaks predictions.
- `GET /Predic_BD` + `POST /search` — bulk prediction: pulls `nombre` random rows from `Db.sqlite3`'s `Client_Bank` table, runs them through the model, and renders actual vs. predicted status side by side.
- `POST /predict_api` — JSON variant of single-client prediction.
- `GET /dashboard` — portfolio-level view, see "Dashboard" below.

`model` is loaded once at import time via `pickle.load`; a bad `model.pkl` or a missing/incompatible ML library fails the whole app at startup, not per-request. `model_metrics.json` (also loaded at import time, optional) feeds the accuracy/recall numbers shown on the home page — if it's missing, that stat line is simply hidden. `EXPLAINER`/`EXPLAINER_KIND` (also built once at import time, see "Explainability" below) back the "Pourquoi ?" reasons on `/predict`'s result pages.

Both `/predict` and `/search` surface risk as a **0-100 Risk Score** plus a 4-tier qualitative label via `risk_level()` (Faible <25, Moyen <50, Élevé <75, Critique ≥75) rather than the model's raw binary class name — deliberately, since "Attrited"/binary framing tested as far less legible to non-technical staff than a score + tier. The binary classifier decision (threshold 0.5) still drives the headline verdict text/icon on `/predict`'s result pages and the `statut` filter on `/search`; note 0.5 sits exactly on the Élevé/Critique boundary, so "predicted to churn" always falls in Élevé or Critique — the two framings can never contradict each other.

### Model training (`train_model.py`)

`model.pkl` is not hand-picked — it's produced by `train_model.py`, which pulls the full `Client_Bank` table, trains five candidates (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost) plus a Stacking ensemble of all five, evaluates each on a held-out test split (accuracy/precision/recall/F1/ROC-AUC + 5-fold CV F1), and picks the winner by **test F1 on the "Attrited" class** (chosen over raw accuracy/recall because the dataset is ~16% attrited — accuracy alone is misleading, and optimizing purely for recall floods the tool with false positives). The winning pipeline is refit on the *entire* dataset before being pickled, so `model.pkl` itself is never evaluated on the held-out split it was scored with.

Re-run it any time `Db.sqlite3` changes or a candidate's hyperparameters are tuned:

```bash
env311/Scripts/python.exe train_model.py
```

This overwrites `model.pkl` and `model_metrics.json`, and writes `model_comparison.csv` (all six candidates' metrics, for the record). Because the winning model type varies by run/data, **whichever library produced the winner must be importable to unpickle `model.pkl`** — that's why `requirements.txt` unconditionally includes xgboost/lightgbm/catboost even though only one of them (or plain scikit-learn) actually backs the model at any given time. If you swap the winner's library out, `app.py`'s `pickle.load` will fail at startup with an `ImportError`, not a subtle runtime bug.

The saved pipeline is always `Pipeline(StandardScaler, <winning estimator>)` over the same 7 features in the same order, so `app.py` needs no changes when the winner changes.

### Explainability (`explain.py`)

`/predict`'s result pages show a "Pourquoi ?" list of the top factors behind a single client's prediction, computed with SHAP. `explain.build_explainer(pipeline)` dispatches on the winning estimator's type:

- **Tree-based** (RandomForest, XGBoost, LightGBM, CatBoost, DecisionTree) → `shap.TreeExplainer` — exact and near-instant (~0.1-0.2s), because it only needs the fitted trees, no background dataset.
- **LogisticRegression** → closed-form `coef_ * scaled_value`, equally instant.
- **Anything else (e.g. Stacking)** → returns `(None, None)`; `explain.explain(...)` then returns `[]` and the templates simply hide the reasons section. This is deliberate: a model-agnostic explainer (KernelExplainer/PermutationExplainer) was measured at ~10s per single prediction here, which is not viable for a live form submission, so an unsupported estimator degrades gracefully rather than stalling the request.

Reasons are ranked by each feature's share of the total SHAP contribution *in the direction of the predicted class* (e.g. only risk-increasing factors are shown for an "at risk" verdict), normalized to sum to ~100%. `Gender` is excluded from the displayed reasons (`explain.HIDDEN_FROM_REASONS`) even though the model uses it — it's not an appropriate or actionable factor to hand a retention agent.

Because this reruns `shap.TreeExplainer`/coefficient math against whatever `model.pkl` currently is, it stays correct across `train_model.py` re-runs with no code changes — as long as the winner is one of the supported types above.

### Dashboard (`dashboard.py`, `templates/Dashboard.html`)

`GET /dashboard` scores the *entire* `Client_Bank` table (not a random sample) and shows: total clients, historical churn rate, average predicted risk, a top-10-by-risk-score table, and churn rate by age bucket / by gender, charted with Chart.js (loaded from CDN, pinned to `chart.js@4.4.4`). `dashboard.build_dashboard_data(model, risk_level_fn)` does the aggregation and returns a plain dict rendered straight into the template — no separate JSON API endpoint.

**Deliberately excluded**: churn by branch, by region, and month-over-month evolution. `Client_Bank` has no branch, region, or date column — those breakdowns would require fabricated data next to real customer records, which is a worse outcome than not having the chart. Only add them once a real data source for those dimensions exists; don't synthesize one.

Bar charts use a single hue (the `--danger` red) since each is one measure (churn rate) across categories, not multiple identity-carrying series — no legend needed. Values are direct-labeled above each bar via a small inline Chart.js plugin rather than a `chartjs-plugin-datalabels` dependency.

### Templates (`templates/`)

All six pages (`index.html`, `Predic_form.html`, `Predic_BD.html`, `Answers0.html`, `Answers1.html`, `Dashboard.html`) extend `templates/_base.html`, which owns the header/nav/footer and pulls in `static/css/design-system.css` + `static/js/app.js`. There is no bundler/build step — CSS and JS are hand-written and served directly by Flask's static handler (Chart.js is the one exception, loaded from CDN only on `Dashboard.html` via its `extra_scripts` block).

`Answers0.html` (stable) / `Answers1.html` (at-risk) are rendered by `/predict` based on the model's binary output; both receive a `prediction_text` string.

**Flask template caching**: the app runs with `debug=False`, so Jinja does not hot-reload `.html` template changes — restart the Flask process after editing a template. Static `.css`/`.js` files *are* served fresh from disk on every request, no restart needed.

### Data (`Db.sqlite3`)

`Client_Bank` table columns used by `/search` and `train_model.py`: `Gender, Customer_Age, Total_Relationship_Count, Months_Inactive_12_mon, Total_Revolving_Bal, Total_Trans_Amt, Avg_Utilization_Ratio, Attrition_Flag`. `Gender` is stored as `'M'`/`'F'` and is mapped to `1`/`0` before being fed to the model (and mapped back for display).

### Design system

`PRODUCT.md` at the repo root captures the product register, target users, and design principles (written via the `impeccable` skill) — read it before making further UI changes. Color tokens, type scale, and component styles live in `static/css/design-system.css` (OKLCH tokens, WCAG AA contrast, `prefers-reduced-motion` respected).
