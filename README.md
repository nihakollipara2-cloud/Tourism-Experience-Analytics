# Tourism Experience Analytics — Complete Project
#Streamlit Link:https://tourism-experience-analytics-z9eqwpu9u5jcm2vntkmfxd.streamlit.app/
## Run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The trained `.joblib` models are already included. Retraining is optional:
```bash
python train_models.py
```

## Structure
- `app.py` — Streamlit application
- `train_models.py` — reproducible model training
- `requirements.txt` — dependencies
- `models/` — trained models and metrics
- `data/processed/` — cleaned data and recommendation tables
- `data/source/` — original supplied Excel files
- `sql_analysis.sql` — SQL analysis queries
- `PROJECT_REPORT.md` — report
- `PRESENTATION_SCRIPT.md` — viva/presentation script

## Main functionality
- Rating prediction
- Visit-mode classification
- Personalized attraction recommendations
- Interactive EDA
- Business insights

## Important modeling choice
Historical features are calculated from visits before each transaction and the evaluation is chronological, reducing target leakage compared with a random split.
