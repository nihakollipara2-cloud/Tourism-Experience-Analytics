# Tourism Experience Analytics — Project Report

## 1. Objective
The project builds a complete tourism analytics system for **rating prediction, visit-mode classification and attraction recommendation**, with an interactive Streamlit interface.

## 2. Supplied Data
The provided dataset contains transaction, user, city, country, region, continent, attraction type, visit mode and attraction tables.

| Table | Rows |
|---|---:|
| Transactions | 52,930 |
| Users | 33,530 |
| Attractions (Updated Item) | 1,698 |
| Cities | 9,143 |
| Countries | 165 |
| Regions | 22 |
| Continents | 6 |
| Attraction Types | 17 |

## 3. Data Preparation
The tables were joined using their ID fields. Missing categorical values were represented as `Unknown`; ratings were limited to 1–5; visit months were converted to numeric month values.

Historical features were created chronologically: previous user visit count, previous user average rating, previous visit-mode counts, previous attraction visit count and previous attraction average rating. A chronological 80/20 train-test split was used.

## 4. EDA
The application visualizes rating distribution, visit modes, geography, time trends, attraction popularity and attraction-type ratings.

## 5. Classification
**Target:** VisitModeName  
**Model:** Random Forest  
**Metrics:**
- Accuracy: 53.55%
- Weighted Precision: 0.599
- Weighted Recall: 0.536
- Weighted F1: 0.550

## 6. Regression
**Target:** Rating  
**Model:** Ridge Regression  
**Metrics:**
- R²: 0.108
- RMSE: 0.952
- MAE: 0.746

The rating task is subjective and the supplied dataset has limited direct preference variables, so the model should be interpreted as an estimate.

## 7. Recommendation System
The recommender ranks attractions using historical user attraction-type preferences, average rating, rating volume, unique-user volume, selected attraction type and selected attraction city. For known users, previously visited attractions are removed.

## 8. Streamlit Modules
1. Dashboard
2. Visit Mode Prediction
3. Rating Prediction
4. Recommendations
5. EDA & Insights
6. About

## 9. Business Value
The system supports personalized discovery, tourism trend analysis, targeted visitor campaigns, attraction performance analysis and customer-experience improvement.

## 10. Limitations and Future Work
Future versions can add map visualization, weather/seasonality, richer attraction descriptions, matrix-factorization collaborative filtering, XGBoost/LightGBM model comparison and SHAP explanations.

## 11. Conclusion
The deliverable provides an end-to-end data science project from raw tourism data through cleaning, EDA, ML, recommendation and Streamlit deployment.
