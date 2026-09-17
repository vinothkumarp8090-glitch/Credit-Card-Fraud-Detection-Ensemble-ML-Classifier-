# Credit Card Fraud Detection — Ensemble ML Classifier

A Flask web application for predicting credit-card default/fraud risk from customer profile, credit-limit, repayment-status, and billing features. The prediction model is trained with XGBoost on the included UCI credit-card dataset.

## Features

- User registration and login backed by SQLite
- Browser form for entering 18 credit-card features
- XGBoost-based binary classification prediction
- Training script with stratified train/test split and ROC-AUC evaluation

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Flask application and prediction routes |
| `model.py` | Trains and saves the XGBoost model |
| `UCI_Credit_Card.csv` | Training dataset |
| `templates/` | HTML pages |
| `static/` | CSS and image assets |
| `create_database.py` | Creates the local `users.db` database |

## Requirements

- Python 3.10 or newer
- pip

Install the Python packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install flask numpy pandas scikit-learn xgboost argon2-cffi gunicorn
```

## Run locally

The trained `credit_model.pkl` file is intentionally not stored in the repository because model artifacts can be large. Generate it from the included dataset before launching the app:

```powershell
python model.py
python app.py
```

Open `http://127.0.0.1:5000` in a browser. On first run, the application automatically creates `users.db` for registered users.

## Train the model

`model.py` uses the `result` column in `UCI_Credit_Card.csv` as the target. It trains an `XGBClassifier`, prints a confusion matrix, classification report, and ROC-AUC score, then writes `credit_model.pkl` to the project root.

## Deployment

The included `Procfile` uses Gunicorn:

```text
web: gunicorn app:app
```

Set a unique Flask secret key through an environment variable before deploying; do not use a development secret key in a public production deployment.

## Notes

- Virtual environments, generated model files (`*.pkl`), caches, and the local SQLite database are excluded with `.gitignore`.
- Do not commit passwords, API keys, or other private data.
