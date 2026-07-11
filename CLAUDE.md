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
pip install flask "scikit-learn==1.2.2" "numpy<2" pandas joblib
```

Do **not** `pip install -r requirements.txt` as-is and expect a working stack — see "model.pkl version pinning" below. Also note `requirements.txt` is UTF-16-encoded (a leftover from being saved by a Windows editor), which trips up naive line-based tooling.

### Run the app

```bash
python wsgi.py          # dev server on http://127.0.0.1:5000, debug=False
```

`Procfile` defines the production command (`gunicorn wsgi:app --preload --workers 3`), but gunicorn doesn't run natively on Windows — use WSL/Linux to test that path.

### No test suite

There are no automated tests in this repository.

## Architecture

### Request flow (`app.py`)

Single-file Flask app, five routes:
- `GET /` — home page.
- `GET /Predic_form` + `POST /predict` — single-client prediction. The form's field order **must** match the model's expected feature order: `Gender, Customer_Age, Total_Relationship_Count, Months_Inactive_12_mon, Total_Revolving_Bal, Total_Trans_Amt, Avg_Utilization_Ratio`. `/predict` builds the feature vector from `request.form.values()` positionally (no field-name mapping), so reordering fields in the template silently breaks predictions.
- `GET /Predic_BD` + `POST /search` — bulk prediction: pulls `nombre` random rows from `Db.sqlite3`'s `Client_Bank` table, runs them through the model, and renders actual vs. predicted status side by side.
- `POST /predict_api` — JSON variant of single-client prediction.

`model` is loaded once at import time via `pickle.load`; a bad `model.pkl` or incompatible scikit-learn version fails the whole app at startup, not per-request.

### model.pkl version pinning (important)

`model.pkl` was pickled under **scikit-learn 0.23.2**. scikit-learn >=1.3 changed the internal `Tree` node dtype (added `missing_go_to_left`), so unpickling raises `ValueError: node array from the pickle has an incompatible dtype`. The working combination confirmed in this environment is **`scikit-learn==1.2.2` + `numpy<2`** (sklearn 1.2.2 is ABI-incompatible with numpy 2.x). Unpickling still emits `InconsistentVersionWarning`/`UserWarning` — that's expected and harmless. Don't "fix" this by upgrading scikit-learn without re-training/re-exporting `model.pkl`.

### Templates (`templates/`)

All five pages (`index.html`, `Predic_form.html`, `Predic_BD.html`, `Answers0.html`, `Answers1.html`) extend `templates/_base.html`, which owns the header/nav/footer and pulls in `static/css/design-system.css` + `static/js/app.js`. There is no bundler/build step — CSS and JS are hand-written and served directly by Flask's static handler.

`Answers0.html` (stable) / `Answers1.html` (at-risk) are rendered by `/predict` based on the model's binary output; both receive a `prediction_text` string.

**Flask template caching**: the app runs with `debug=False`, so Jinja does not hot-reload `.html` template changes — restart the Flask process after editing a template. Static `.css`/`.js` files *are* served fresh from disk on every request, no restart needed.

### Data (`Db.sqlite3`)

`Client_Bank` table columns used by `/search`: `Gender, Customer_Age, Total_Relationship_Count, Months_Inactive_12_mon, Total_Revolving_Bal, Total_Trans_Amt, Avg_Utilization_Ratio, Attrition_Flag`. `Gender` is stored as `'M'`/`'F'` and is mapped to `1`/`0` before being fed to the model (and mapped back for display). `Db.db` is an unrelated empty file — not used by the app.

### Design system

`PRODUCT.md` at the repo root captures the product register, target users, and design principles (written via the `impeccable` skill) — read it before making further UI changes. Color tokens, type scale, and component styles live in `static/css/design-system.css` (OKLCH tokens, WCAG AA contrast, `prefers-reduced-motion` respected).
