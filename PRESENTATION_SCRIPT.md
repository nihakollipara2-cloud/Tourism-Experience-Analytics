# Presentation Script — Tourism Experience Analytics
## Slide 1 — Title
Our project is Tourism Experience Analytics: Classification, Prediction, and Recommendation System. We use tourism data to understand visitor behavior and improve attraction discovery.

## Slide 2 — Problem
Tourism platforms need to understand visitor patterns, estimate satisfaction and recommend relevant attractions. We address these with classification, regression and recommendation.

## Slide 3 — Dataset
The supplied dataset includes transactions, users, cities, countries, regions, continents, attraction types, visit modes and attraction details. We join these tables using IDs.

## Slide 4 — Data Preparation
We clean missing values, standardize months and ratings, and create historical features such as previous user ratings and previous visit-mode counts.

## Slide 5 — EDA
We analyze rating distributions, visit modes, geography, time trends, attraction popularity and attraction-type performance.

## Slide 6 — Classification
We predict Business, Couples, Family, Friends or Solo using a Random Forest model.

## Slide 7 — Regression
We predict a rating from 1 to 5 using Ridge Regression and demographic, temporal, attraction and historical features.

## Slide 8 — Recommendation
Our recommender uses user history when available and combines personal preferences with attraction popularity and selected location/type.

## Slide 9 — Streamlit
The application provides a dashboard, two prediction modules, recommendations, EDA and project documentation in one interface.

## Slide 10 — Evaluation
Classification uses accuracy, precision, recall and F1. Regression uses R², RMSE and MAE. We use a chronological 80/20 split.

## Slide 11 — Business Value
The solution can support personalized tourism experiences, targeted marketing and attraction performance decisions.

## Slide 12 — Conclusion
This project demonstrates a complete data science workflow from raw tourism data to an interactive Streamlit application.
