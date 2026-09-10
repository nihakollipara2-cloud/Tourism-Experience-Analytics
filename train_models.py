# Re-train models from data/processed/tourism_cleaned.csv. The generated features are already chronological.
# Run: python train_models.py
import os,json,joblib,numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,r2_score,mean_squared_error,mean_absolute_error
B=os.path.dirname(os.path.abspath(__file__)); d=pd.read_csv(os.path.join(B,"data","processed","tourism_cleaned.csv")); cut=int(len(d)*.8); tr=d.iloc[:cut]; te=d.iloc[cut:]
CLS=["Continent","UserRegion","UserCountry","UserCityName","AttractionType","Attraction","VisitYear","VisitMonthNum","user_prev_count","user_prev_avg_rating","user_prev_mode_1","user_prev_mode_2","user_prev_mode_3","user_prev_mode_4","user_prev_mode_5","attr_prev_count","attr_prev_avg_rating"]
REG=["Continent","UserRegion","UserCountry","UserCityName","AttractionType","Attraction","VisitYear","VisitMonthNum","VisitModeName","user_prev_count","user_prev_avg_rating","attr_prev_count","attr_prev_avg_rating"]
def prep(f):
 c=[x for x in f if d[x].dtype=="object"]; n=[x for x in f if x not in c]
 return ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),c),("num",SimpleImputer(strategy="median"),n)])
clf=Pipeline([("preprocess",prep(CLS)),("model",RandomForestClassifier(n_estimators=180,max_depth=22,min_samples_leaf=2,class_weight="balanced_subsample",random_state=42,n_jobs=-1))]); clf.fit(tr[CLS],tr.VisitModeName); cp=clf.predict(te[CLS])
reg=Pipeline([("preprocess",prep(REG)),("model",Ridge(alpha=10.0))]); reg.fit(tr[REG],tr.Rating); rp=np.clip(reg.predict(te[REG]),1,5)
m={"classification":{"accuracy":accuracy_score(te.VisitModeName,cp),"precision_weighted":precision_score(te.VisitModeName,cp,average="weighted",zero_division=0),"recall_weighted":recall_score(te.VisitModeName,cp,average="weighted",zero_division=0),"f1_weighted":f1_score(te.VisitModeName,cp,average="weighted",zero_division=0)},"regression":{"r2":r2_score(te.Rating,rp),"rmse":np.sqrt(mean_squared_error(te.Rating,rp)),"mae":mean_absolute_error(te.Rating,rp)}}
os.makedirs(os.path.join(B,"models"),exist_ok=True); joblib.dump(clf,os.path.join(B,"models","visit_mode_classifier.joblib")); joblib.dump(reg,os.path.join(B,"models","rating_regressor.joblib")); json.dump(m,open(os.path.join(B,"models","metrics.json"),"w"),indent=2); print(json.dumps(m,indent=2))
