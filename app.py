import os,json,joblib,numpy as np,pandas as pd,streamlit as st
st.set_page_config(page_title="Tourism Experience Analytics",page_icon="✈️",layout="wide")
BASE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(BASE,"data","processed"); MODELS=os.path.join(BASE,"models")
@st.cache_data
def load_data(): return pd.read_csv(os.path.join(DATA,"tourism_cleaned.csv")),pd.read_csv(os.path.join(DATA,"attraction_profiles.csv"))
@st.cache_resource
def load_models():
    c=joblib.load(os.path.join(MODELS,"visit_mode_classifier.joblib")); r=joblib.load(os.path.join(MODELS,"rating_regressor.joblib"))
    with open(os.path.join(MODELS,"metrics.json")) as f: m=json.load(f)
    return c,r,m
tourism,attractions=load_data(); classifier,regressor,metrics=load_models()
def user_hist(uid):
    h=tourism[tourism.UserId==uid] if uid is not None else tourism.iloc[0:0]
    v=h.VisitMode.value_counts().to_dict()
    return {"count":len(h),"avg":float(h.Rating.mean()) if len(h) else float(tourism.Rating.mean()),"modes":{i:int(v.get(i,0)) for i in range(1,6)}}
def attr_hist(aid):
    h=tourism[tourism.AttractionId==aid]
    return {"count":len(h),"avg":float(h.Rating.mean()) if len(h) else float(tourism.Rating.mean())}
def selected_attr_id(name):
    x=attractions[attractions.Attraction==name]
    return int(x.iloc[0].AttractionId) if len(x) else None
def cls_row(vals):
    c,rg,co,city,typ,att,yr,mo,uh,ah=vals
    return pd.DataFrame([{"Continent":c,"UserRegion":rg,"UserCountry":co,"UserCityName":city,"AttractionType":typ,"Attraction":att,"VisitYear":yr,"VisitMonthNum":mo,"user_prev_count":uh["count"],"user_prev_avg_rating":uh["avg"],**{f"user_prev_mode_{i}":uh["modes"][i] for i in range(1,6)},"attr_prev_count":ah["count"],"attr_prev_avg_rating":ah["avg"]}])
def reg_row(vals):
    c,rg,co,city,typ,att,yr,mo,vm,uh,ah=vals
    return pd.DataFrame([{"Continent":c,"UserRegion":rg,"UserCountry":co,"UserCityName":city,"AttractionType":typ,"Attraction":att,"VisitYear":yr,"VisitMonthNum":mo,"VisitModeName":vm,"user_prev_count":uh["count"],"user_prev_avg_rating":uh["avg"],"attr_prev_count":ah["count"],"attr_prev_avg_rating":ah["avg"]}])
def controls(title,with_user=True):
    st.subheader(title)
    uids=[None]+sorted(tourism.UserId.dropna().astype(int).unique().tolist())
    uid=st.selectbox("Existing User (optional)",uids,format_func=lambda x:"New / no history" if x is None else str(x)) if with_user else None
    c1,c2,c3=st.columns(3)
    cont=c1.selectbox("Continent",sorted(tourism.Continent.unique()))
    region=c2.selectbox("Region",sorted(tourism.UserRegion.unique()))
    country=c3.selectbox("Country",sorted(tourism.UserCountry.unique()))
    c4,c5,c6=st.columns(3)
    city=c4.selectbox("User City",sorted(tourism.UserCityName.unique()))
    typ=c5.selectbox("Attraction Type",sorted(tourism.AttractionType.unique()))
    att=c6.selectbox("Attraction",sorted(tourism.Attraction.unique()))
    yr=st.number_input("Visit Year",int(tourism.VisitYear.min()),int(tourism.VisitYear.max()+2),int(tourism.VisitYear.max()))
    months=["January","February","March","April","May","June","July","August","September","October","November","December"]
    mo_name=st.selectbox("Visit Month",months); mo=months.index(mo_name)+1
    return uid,cont,region,country,city,typ,att,yr,mo
page=st.sidebar.radio("Navigate",["Dashboard","Visit Mode Prediction","Rating Prediction","Recommendations","EDA & Insights","About"])
if page=="Dashboard":
    st.title("✈️ Tourism Experience Analytics"); st.caption("Classification • Rating Prediction • Personalized Recommendations")
    a,b,c,d=st.columns(4); a.metric("Transactions",f"{len(tourism):,}"); b.metric("Users",f"{tourism.UserId.nunique():,}"); c.metric("Attractions",f"{attractions.AttractionId.nunique():,}"); d.metric("Average Rating",f"{tourism.Rating.mean():.2f}/5")
    l,r=st.columns(2)
    with l: st.write("**Visit Mode Distribution**"); st.bar_chart(tourism.VisitModeName.value_counts())
    with r: st.write("**Top Attraction Types by Average Rating**"); st.bar_chart(tourism.groupby("AttractionType").Rating.mean().sort_values(ascending=False).head(10))
elif page=="Visit Mode Prediction":
    st.title("🧭 Visit Mode Prediction"); st.write("Predict Business, Couples, Family, Friends or Solo.")
    uid,c,rg,co,city,typ,att,yr,mo=controls("Visitor and attraction details")
    if st.button("Predict Visit Mode",type="primary"):
        uh=user_hist(uid); aid=selected_attr_id(att); ah=attr_hist(aid) if aid else {"count":0,"avg":tourism.Rating.mean()}
        row=cls_row((c,rg,co,city,typ,att,yr,mo,uh,ah)); p=classifier.predict(row)[0]; pr=classifier.predict_proba(row)[0]
        i=int(np.argmax(pr)); st.success(f"Predicted visit mode: **{p}**"); st.metric("Confidence",f"{pr[i]*100:.1f}%")
        st.dataframe(pd.DataFrame({"Visit Mode":classifier.classes_,"Probability (%)":(pr*100).round(2)}).sort_values("Probability (%)",ascending=False),hide_index=True,use_container_width=True)
elif page=="Rating Prediction":
    st.title("⭐ Attraction Rating Prediction")
    uid,c,rg,co,city,typ,att,yr,mo=controls("Visitor and attraction details")
    vm=st.selectbox("Visit Mode",["Business","Couples","Family","Friends","Solo"])
    if st.button("Predict Rating",type="primary"):
        uh=user_hist(uid); aid=selected_attr_id(att); ah=attr_hist(aid) if aid else {"count":0,"avg":tourism.Rating.mean()}
        rating=float(np.clip(regressor.predict(reg_row((c,rg,co,city,typ,att,yr,mo,vm,uh,ah)))[0],1,5))
        st.success(f"Predicted rating: **{rating:.2f} / 5**"); st.progress(rating/5)
elif page=="Recommendations":
    st.title("🗺️ Personalized Attraction Recommendations")
    uids=[None]+sorted(tourism.UserId.dropna().astype(int).unique().tolist())
    uid=st.selectbox("User (optional)",uids,format_func=lambda x:"New / no history" if x is None else str(x))
    typ=st.selectbox("Preferred Attraction Type",["All"]+sorted(attractions.AttractionType.unique()))
    city_id=st.selectbox("Preferred Attraction City ID",["All"]+sorted(attractions.AttractionCityId.dropna().astype(int).astype(str).unique()))
    n=st.slider("Number of recommendations",5,20,10)
    if st.button("Generate Recommendations",type="primary"):
        p=attractions.copy()
        if uid is not None: p=p[~p.AttractionId.isin(set(tourism.loc[tourism.UserId==uid,"AttractionId"].astype(int)))]
        p["type_match"]=(p.AttractionType==typ).astype(int) if typ!="All" else 0
        p["city_match"]=(p.AttractionCityId==int(city_id)).astype(int) if city_id!="All" else 0
        h=tourism[tourism.UserId==uid] if uid is not None else tourism.iloc[0:0]
        prefs=h.groupby("AttractionType").Rating.mean().to_dict()
        p["personal"]=p.AttractionType.map(prefs).fillna(p.AvgRating)
        p["score"]=.50*p.personal+.25*p.AvgRating+.15*p.type_match+.10*p.city_match
        p=p.sort_values(["score","RatingCount"],ascending=False).head(n)
        out=p[["Attraction","AttractionType","AttractionAddress","AvgRating","RatingCount","score"]].copy()
        out.columns=["Attraction","Type","Address","Avg Rating","Ratings","Recommendation Score"]; out[["Avg Rating","Recommendation Score"]]=out[["Avg Rating","Recommendation Score"]].round(3)
        st.dataframe(out,hide_index=True,use_container_width=True)
elif page=="EDA & Insights":
    st.title("📊 EDA & Insights")
    t1,t2,t3,t4=st.tabs(["Ratings","Geography","Trends","Attractions"])
    with t1: st.bar_chart(tourism.Rating.value_counts().sort_index())
    with t2:
        st.write("Top countries"); st.bar_chart(tourism.UserCountry.value_counts().head(15)); st.write("Continents"); st.bar_chart(tourism.Continent.value_counts())
    with t3:
        q=tourism.groupby(["VisitYear","VisitMonthNum"]).size().reset_index(name="Visits"); q["Period"]=q.VisitYear.astype(str)+"-"+q.VisitMonthNum.astype(str).str.zfill(2); st.line_chart(q.set_index("Period").Visits)
    with t4: st.dataframe(attractions.sort_values(["RatingCount","AvgRating"],ascending=False).head(20),hide_index=True,use_container_width=True)
    st.subheader("Model Performance")
    a,b,c,d=st.columns(4); a.metric("Accuracy",f"{metrics['classification']['accuracy']*100:.2f}%"); b.metric("Weighted F1",f"{metrics['classification']['f1_weighted']:.3f}"); c.metric("R²",f"{metrics['regression']['r2']:.3f}"); d.metric("RMSE",f"{metrics['regression']['rmse']:.3f}")
else:
    st.title("ℹ️ About")
    st.markdown("""### Tourism Experience Analytics
**Classification, Prediction, and Recommendation System**

**Modules:** regression for rating prediction, classification for visit mode prediction, hybrid attraction recommendation, EDA and business insights.

**Models:** Random Forest classifier and Ridge regression. Categorical variables use one-hot encoding. Historical user/attraction features are calculated from visits before each transaction. Evaluation uses a chronological 80/20 split.

This is an academic analytics application; predictions are estimates rather than guaranteed behavior.
""")
