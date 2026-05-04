"""
HMDA Fair Lending Analysis Dashboard
Washington DC MSA · 2023
Phase 5 — Streamlit + Folium + XGBoost SHAP + Gemini AI Analyst
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os, pickle, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="HMDA Fair Lending · DC MSA 2023",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Epilogue:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Epilogue', sans-serif; }
.stApp { background: #0c0e14; color: #dde1ed; }
[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1e2130; }
[data-testid="stSidebar"] * { color: #dde1ed !important; }
.dash-banner {
    background: linear-gradient(135deg,#111318 0%,#161923 100%);
    border:1px solid #1e2130; border-radius:12px;
    padding:1.5rem 2rem; margin-bottom:1.5rem; position:relative; overflow:hidden;
}
.dash-banner::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#f5a623,#1ec9a0,#4a9eff);
}
.dash-title { font-size:1.6rem; font-weight:800; letter-spacing:-0.03em; color:#f0f4ff; margin:0; }
.dash-sub { font-family:'DM Mono',monospace; font-size:0.7rem; color:#6b7491; margin-top:6px; letter-spacing:0.08em; text-transform:uppercase; }
.dash-rq { font-size:0.8rem; color:#8891aa; margin-top:8px; font-style:italic; border-left:2px solid #f5a623; padding-left:10px; }
.kpi-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-bottom:1.5rem; }
.kpi { background:#111318; border:1px solid #1e2130; border-radius:10px; padding:1rem 1.1rem; text-align:center; position:relative; overflow:hidden; }
.kpi::after { content:''; position:absolute; bottom:0; left:0; right:0; height:2px; }
.kpi.red::after{background:#e05555;} .kpi.amber::after{background:#f5a623;} .kpi.teal::after{background:#1ec9a0;} .kpi.blue::after{background:#4a9eff;}
.kpi-val { font-family:'DM Mono',monospace; font-size:1.8rem; font-weight:500; line-height:1; }
.kpi-val.red{color:#e05555;} .kpi-val.amber{color:#f5a623;} .kpi-val.teal{color:#1ec9a0;} .kpi-val.blue{color:#4a9eff;}
.kpi-lbl { font-size:0.65rem; color:#6b7491; margin-top:6px; text-transform:uppercase; letter-spacing:0.08em; line-height:1.4; }
.sec-label { font-size:0.65rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; color:#6b7491; border-bottom:1px solid #1e2130; padding-bottom:6px; margin-bottom:1rem; }
.finding { background:#111318; border-left:3px solid #1ec9a0; border-radius:0 8px 8px 0; padding:0.75rem 1rem; margin:0.5rem 0; font-size:0.82rem; line-height:1.7; color:#b8c0d8; }
.finding.warn{border-color:#f5a623;} .finding.danger{border-color:#e05555;} .finding.info{border-color:#4a9eff;}
.ai-bubble { background:#111318; border:1px solid #1e2130; border-radius:12px 12px 12px 0; padding:1rem 1.2rem; margin:0.5rem 0; font-size:0.85rem; line-height:1.75; color:#c8d0e0; }
.user-bubble { background:#161923; border:1px solid #252a3a; border-radius:12px 12px 0 12px; padding:0.65rem 1rem; margin:0.5rem 0; font-size:0.85rem; color:#dde1ed; text-align:right; }
.ai-label { font-family:'DM Mono',monospace; font-size:0.65rem; color:#1ec9a0; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:4px; }
.user-label { font-family:'DM Mono',monospace; font-size:0.65rem; color:#6b7491; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:4px; text-align:right; }
.stSelectbox>div>div, .stTextInput>div>div { background:#161923 !important; border-color:#1e2130 !important; color:#dde1ed !important; }
.stButton>button { background:#161923; border:1px solid #252a3a; color:#dde1ed; border-radius:8px; font-family:'DM Mono',monospace; font-size:0.75rem; letter-spacing:0.05em; }
.stButton>button:hover { border-color:#f5a623; color:#f5a623; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:1.5rem; padding-bottom:2rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    hmda        = pd.read_csv("data/hmda_dashboard.csv", low_memory=False)
    tract_stats = pd.read_csv("data/tract_stats.csv")
    lenders     = pd.read_csv("data/lender_disparity.csv")
    with open("data/key_stats.json")      as f: stats    = json.load(f)
    with open("data/model_features.json") as f: features = json.load(f)
    return hmda, tract_stats, lenders, stats, features

@st.cache_data
def load_geo():
    import geopandas as gpd
    gdf = gpd.read_file("data/tracts.geojson")

    # Normalise index — geopandas sometimes promotes GEOID to index
    gdf = gdf.reset_index()

    # Find the GEOID column regardless of what it is named
    if "GEOID" not in gdf.columns:
        for col in gdf.columns:
            if col in ("geometry", "index"): continue
            sample = gdf[col].dropna().astype(str)
            if len(sample) > 0 and sample.iloc[0].isdigit() and len(sample.iloc[0]) >= 10:
                gdf = gdf.rename(columns={col: "GEOID"})
                break

    gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(11)
    return gdf

@st.cache_resource
def load_model():
    with open("data/xgb_model.pkl","rb") as f: return pickle.load(f)

hmda, tract_stats, lenders, stats, model_features = load_data()
tracts_geo = load_geo()
model      = load_model()

# Normalise tract_stats census_tract key to match tracts_geo GEOID format
if "GEOID" not in tract_stats.columns:
    if "census_tract" in tract_stats.columns:
        tract_stats = tract_stats.rename(columns={"census_tract": "GEOID"})
tract_stats["GEOID"] = tract_stats["GEOID"].astype(str).str.zfill(11)


# ── Gemini ────────────────────────────────────────────────────────────────────
def get_gemini():
    try:
        import google.generativeai as genai
        key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY",""))
        if not key: return None
        genai.configure(api_key=key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception: return None

gemini = get_gemini()

SYSTEM_CTX = f"""You are a senior fair lending analyst specializing in HMDA analysis and ECOA enforcement.

Analyzing 2023 HMDA home purchase mortgage data for the Washington DC MSA (47894).

KEY FINDINGS:
• Total applications: {stats['total_applications']:,} | Overall approval: {stats['overall_approval_rate']:.1%}
• White approval: {stats['white_approval_rate']:.1%} | Black: {stats['black_approval_rate']:.1%} | Hispanic: {stats['hispanic_approval_rate']:.1%}
• Black-White gap: {stats['bw_gap_pp']:.1f}pp
• Black applicants: higher DTI (44% vs 39%), LTI (3.40x vs 3.05x), LTV (96.5% vs 85.0%)
• 26.4% of denials cite NO financial reason
• Gap significant in ALL 4 DTI bands (p=0.0000). At DTI ≤36%: 7.9pp gap, p=0.0000
• CFPB disparity index: Black={stats['black_disparity_index']:.2f}x, Hispanic={stats['hisp_disparity_index']:.2f}x (>2.0x = significant)
• Moran's I={stats['morans_i']:.3f} (p={stats['morans_p']:.3f}): HH tracts {stats['hh_minority_pct']}% minority vs LL tracts {stats['ll_minority_pct']}%
• 76% of lenders exceed 2.0x threshold | Median ratio: {stats['median_disparity_ratio']:.2f}x
• Top lender: {stats['top_lender_name']} at {stats['top_lender_ratio']:.2f}x | JPMorgan Chase: 6.88x
• Logistic regression: Black OR={stats['black_odds_ratio']:.3f} (33.5% lower odds), Hispanic OR=0.786, Asian OR=0.830
• XGBoost AUC={stats['xgb_auc']:.4f} | Race (Black) ranks #2 of 13 features by SHAP
• Regulatory: ECOA (15 U.S.C. § 1691), Fair Housing Act, CRA, CFPB thresholds 1.5x/2.0x

Answer concisely with specific numbers. Flag regulatory implications. Keep under 200 words unless detail requested."""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sec-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["🗺️  Map","📊  Disparities","🏦  Lenders","🤖  SHAP Model","💬  AI Analyst"],
                    label_visibility="collapsed")
    st.markdown('<div class="sec-label" style="margin-top:1.5rem">Filters</div>', unsafe_allow_html=True)
    COUNTIES = {
        "All Counties":None, "Washington DC":"11001",
        "Prince George's County MD":"24033", "Montgomery County MD":"24031",
        "Fairfax County VA":"51059", "Arlington County VA":"51013",
        "Loudoun County VA":"51107", "Prince William County VA":"51153",
    }
    county_sel  = st.selectbox("County", list(COUNTIES.keys()))
    county_fips = COUNTIES[county_sel]
    min_apps    = st.slider("Min applications per tract", 10, 100, 10, 5)
    st.markdown("""<div style="font-size:0.72rem;color:#6b7491;line-height:1.8;margin-top:1rem">
    <b style="color:#b8c0d8">Data:</b> FFIEC HMDA 2023<br>
    <b style="color:#b8c0d8">MSA:</b> Washington DC (47894)<br>
    <b style="color:#b8c0d8">Type:</b> Home purchase loans<br>
    <b style="color:#b8c0d8">Apps:</b> 59,479 | <b style="color:#b8c0d8">Tracts:</b> 1,095<br>
    <i>Research only. Not legal advice.</i></div>""", unsafe_allow_html=True)


# ── Header + KPIs ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dash-banner">
  <div class="dash-title">HMDA Fair Lending Analysis</div>
  <div class="dash-sub">Washington DC MSA · 2023 · Home Purchase Loans · {stats['total_applications']:,} Applications</div>
  <div class="dash-rq">Do location and race affect mortgage approval, even when controlling for financial characteristics?</div>
</div>
<div class="kpi-grid">
  <div class="kpi red"><div class="kpi-val red">{stats['bw_gap_pp']:.1f}pp</div><div class="kpi-lbl">Black-White<br>approval gap</div></div>
  <div class="kpi red"><div class="kpi-val red">{stats['black_disparity_index']:.2f}x</div><div class="kpi-lbl">CFPB disparity<br>index (Black)</div></div>
  <div class="kpi amber"><div class="kpi-val amber">{stats['morans_i']:.3f}</div><div class="kpi-lbl">Moran's I<br>spatial clustering</div></div>
  <div class="kpi amber"><div class="kpi-val amber">{stats['black_odds_ratio']:.3f}</div><div class="kpi-lbl">Black approval<br>odds ratio</div></div>
  <div class="kpi teal"><div class="kpi-val teal">{stats['xgb_auc']:.3f}</div><div class="kpi-lbl">XGBoost<br>AUC-ROC</div></div>
</div>
""", unsafe_allow_html=True)


# ── Page: Map ─────────────────────────────────────────────────────────────────
if page == "🗺️  Map":
    import folium
    from streamlit_folium import st_folium

    st.markdown('<div class="sec-label">Interactive Denial Rate Map</div>', unsafe_allow_html=True)
    col_ctrl, col_map = st.columns([1, 3])

    with col_ctrl:
        metric = st.radio("Color tracts by",["Denial Rate","Minority Pop %","Approval Rate"])
        mcol   = {"Denial Rate":"denial_rate","Minority Pop %":"minority_pct","Approval Rate":"approval_rate"}[metric]
        st.markdown("""
        <div class="finding danger">HH hotspot tracts: median <b>95.2%</b> minority pop.<br>LL coldspot tracts: <b>37.1%</b> minority pop.</div>
        <div class="finding warn" style="margin-top:.5rem">Moran's I=0.267 (p=0.001, z=14.7). Denial clustering is <b>not random</b>.</div>
        """, unsafe_allow_html=True)

    with col_map:
        # Merge tracts_geo with tract_stats — handle both possible key names
        ts = tract_stats.copy()
        if "GEOID" not in ts.columns and "census_tract" in ts.columns:
            ts = ts.rename(columns={"census_tract": "GEOID"})
        ts["GEOID"] = ts["GEOID"].astype(str).str.zfill(11)

        pl = tracts_geo.merge(ts, on="GEOID", how="left")
        pl = pl[pl["total_applications"].notna()].copy()
        pl = pl[pl["total_applications"] >= min_apps].copy()
        if county_fips: pl = pl[pl["GEOID"].str[:5]==county_fips]
        if "approval_rate" not in pl.columns: pl["approval_rate"] = 1 - pl["denial_rate"]

        center = [38.95,-77.05]; zoom = 10
        if county_fips and len(pl)>0:
            b=pl.total_bounds; center=[(b[1]+b[3])/2,(b[0]+b[2])/2]; zoom=11

        m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB dark_matter")
        cmap = "YlOrRd" if mcol=="denial_rate" else "RdYlGn" if mcol=="approval_rate" else "YlOrBr"
        folium.Choropleth(
            geo_data=pl.__geo_interface__, data=pl,
            columns=["GEOID",mcol], key_on="feature.properties.GEOID",
            fill_color=cmap, fill_opacity=0.75, line_opacity=0.15,
            legend_name=metric, nan_fill_color="#2a2f3e",
        ).add_to(m)
        folium.GeoJson(pl, style_function=lambda x:{"fillOpacity":0,"weight":0},
            tooltip=folium.GeoJsonTooltip(
                fields=["GEOID","denial_rate","total_applications","minority_pct"],
                aliases=["Tract","Denial Rate","Applications","Minority Pop %"],
                localize=True,
                style="background:#161923;color:#dde1ed;border:1px solid #252a3a;border-radius:6px;font-family:'DM Mono',monospace;font-size:12px;padding:8px;"
            )).add_to(m)
        st_folium(m, width=None, height=530, returned_objects=[])


# ── Page: Disparities ─────────────────────────────────────────────────────────
elif page == "📊  Disparities":
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown('<div class="sec-label">Racial Approval Gap Analysis</div>', unsafe_allow_html=True)
    RACES  = ["White","Black or African American","Asian","Hispanic or Latino"]
    COLORS = {"White":"#2ecc71","Black or African American":"#e74c3c","Asian":"#3498db","Hispanic or Latino":"#f39c12"}
    LABELS = {r:r.replace(" or African American","").replace(" or Latino","") for r in RACES}

    hf = hmda[hmda["race_ethnicity"].isin(RACES)].copy()
    if county_fips:
        hf = hf[hf["county_code"].astype(str).str.split(".").str[0].str.zfill(5)==county_fips]

    ca, cb = st.columns(2)
    with ca:
        rs = hf.groupby("race_ethnicity")["approved"].agg(n="count",rate="mean").reset_index()
        rs["label"]=rs["race_ethnicity"].map(LABELS); rs["color"]=rs["race_ethnicity"].map(COLORS)
        rs=rs.sort_values("rate",ascending=True)
        wr=rs.loc[rs["race_ethnicity"]=="White","rate"].values[0]
        fig=go.Figure(go.Bar(x=rs["rate"]*100,y=rs["label"],orientation="h",marker_color=rs["color"],
            opacity=0.85,text=rs["rate"].map("{:.1%}".format),textposition="outside"))
        fig.add_vline(x=wr*100,line_dash="dash",line_color="#2ecc71",line_width=1.5,
            annotation_text="White baseline",annotation_font_color="#2ecc71")
        fig.update_layout(title="Approval Rate by Race/Ethnicity",xaxis_title="Approval Rate (%)",
            xaxis=dict(range=[75,100]),paper_bgcolor="#111318",plot_bgcolor="#111318",
            font=dict(color="#dde1ed",family="Epilogue"),title_font_size=13,height=300,
            margin=dict(l=10,r=70,t=40,b=40))
        st.plotly_chart(fig,use_container_width=True)

    with cb:
        rows=[]
        for race in RACES:
            for lo,hi,lbl in [(0,36,"≤36%"),(36,43,"37–43%"),(43,50,"44–50%"),(50,100,">50%")]:
                sub=hf[(hf["race_ethnicity"]==race)&(hf["dti_numeric"]>lo)&(hf["dti_numeric"]<=hi)]
                if len(sub)>=30: rows.append({"race":LABELS[race],"band":lbl,"rate":sub["approved"].mean()*100,"color":COLORS[race]})
        fig2=px.bar(pd.DataFrame(rows),x="band",y="rate",color="race",barmode="group",
            color_discrete_map={LABELS[r]:c for r,c in COLORS.items()},
            labels={"rate":"Approval Rate (%)","band":"DTI Band","race":"Race"},
            title="Approval Rate by DTI Band × Race")
        fig2.update_layout(paper_bgcolor="#111318",plot_bgcolor="#111318",
            font=dict(color="#dde1ed",family="Epilogue"),title_font_size=13,height=300,
            margin=dict(l=10,r=10,t=40,b=40),legend=dict(bgcolor="#111318"))
        fig2.update_yaxes(range=[60,100])
        st.plotly_chart(fig2,use_container_width=True)

    st.markdown('<div class="sec-label">Key Statistical Findings</div>', unsafe_allow_html=True)
    f1,f2,f3=st.columns(3)
    with f1: st.markdown('<div class="finding danger"><b>10.3pp Black-White gap</b> — significant at p&lt;0.0001 in all 4 DTI bands. At DTI ≤36%: <b>7.9pp gap, p=0.0000</b>. No financial argument survives this.</div>', unsafe_allow_html=True)
    with f2: st.markdown('<div class="finding warn"><b>CFPB disparity index:</b> Black <b>2.79x</b>, Hispanic <b>2.42x</b>. Both exceed the 2.0x enforcement threshold. 76% of lenders flagged.</div>', unsafe_allow_html=True)
    with f3: st.markdown('<div class="finding info"><b>Logistic regression OR=0.665:</b> Black applicants have <b>33.5% lower approval odds</b> vs White, controlling for all financial variables simultaneously.</div>', unsafe_allow_html=True)


# ── Page: Lenders ─────────────────────────────────────────────────────────────
elif page == "🏦  Lenders":
    import plotly.graph_objects as go

    st.markdown('<div class="sec-label">Lender-Level Disparity Ranking</div>', unsafe_allow_html=True)
    cc,ct=st.columns([3,2])

    with cc:
        n=st.slider("Show top N lenders",5,20,10)
        top=lenders.nlargest(n,"disparity_ratio").copy()
        top["name_short"]=top["lender_name"].str[:42]
        top["color"]=top["disparity_ratio"].apply(lambda x:"#e05555" if x>2.0 else "#f5a623" if x>1.5 else "#4a9eff")
        fig=go.Figure(go.Bar(x=top["disparity_ratio"],y=top["name_short"],orientation="h",
            marker_color=top["color"],opacity=0.88,
            text=top["disparity_ratio"].map("{:.2f}x".format),textposition="outside",
            customdata=np.stack([top["w_denial_rate"]*100,top["b_denial_rate"]*100,top["denial_gap_pp"],top["w_apps"],top["b_apps"]],axis=-1),
            hovertemplate="<b>%{y}</b><br>White: %{customdata[0]:.1f}%<br>Black: %{customdata[1]:.1f}%<br>Gap: %{customdata[2]:+.1f}pp<br>W apps: %{customdata[3]:,}<br>B apps: %{customdata[4]:,}<extra></extra>"))
        fig.add_vline(x=1.0,line_color="#dde1ed",line_width=0.8,annotation_text="Parity",annotation_font_color="#dde1ed")
        fig.add_vline(x=1.5,line_color="#f5a623",line_dash="dash",line_width=1.2,annotation_text="CFPB flag 1.5x",annotation_font_color="#f5a623")
        fig.add_vline(x=2.0,line_color="#e05555",line_dash="dash",line_width=1.5,annotation_text="Significant 2.0x",annotation_font_color="#e05555")
        fig.update_layout(title=f"Top {n} Lenders — Black÷White Denial Rate Ratio",xaxis_title="Disparity Ratio",
            paper_bgcolor="#111318",plot_bgcolor="#111318",font=dict(color="#dde1ed",family="Epilogue"),
            title_font_size=13,height=max(360,n*40),margin=dict(l=10,r=80,t=40,b=40))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig,use_container_width=True)

    with ct:
        st.markdown('<div class="finding danger"><b>76% of lenders</b> exceed CFPB\'s 2.0x threshold.<br>Median ratio: <b>2.75x</b> across 45 lenders.<br><br>This is a <b>market-wide structural pattern</b>, not isolated bad actors.</div>', unsafe_allow_html=True)
        d=top[["lender_name","w_denial_rate","b_denial_rate","disparity_ratio"]].copy()
        d.columns=["Institution","White","Black","Ratio"]
        d["White"]=d["White"].map("{:.1%}".format); d["Black"]=d["Black"].map("{:.1%}".format)
        d["Ratio"]=d["Ratio"].map("{:.2f}x".format); d["Institution"]=d["Institution"].str[:32]
        st.dataframe(d,use_container_width=True,hide_index=True,height=380)


# ── Page: SHAP ────────────────────────────────────────────────────────────────
elif page == "🤖  SHAP Model":
    import plotly.graph_objects as go

    st.markdown('<div class="sec-label">XGBoost + SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="finding info"><b>Model:</b> XGBoost · AUC=0.800 · 13 features · interest_rate excluded (reporting artifact). <b>Race (Black) ranks #2 of 13</b> by SHAP importance — behind only DTI.</div>', unsafe_allow_html=True)

    cs,cp=st.columns([1,1])
    with cs:
        fi=[("dti_numeric",0.6194,"fin"),("race: Black",0.2320,"race"),("loan_type",0.2001,"fin"),
            ("loan_amount",0.1850,"fin"),("loan_to_value_ratio",0.1669,"fin"),("tract minority %",0.1373,"geo"),
            ("race: Asian",0.1000,"race"),("race: Hispanic",0.0811,"race"),("lti",0.0786,"fin"),
            ("income",0.0735,"fin"),("lien_status",0.0694,"fin"),("loan_term",0.0601,"fin"),("tract income %",0.0596,"geo")]
        CM={"race":"#e05555","geo":"#4a9eff","fin":"#6b7491"}
        fi_s=sorted(fi,key=lambda x:x[1])
        fig=go.Figure(go.Bar(x=[v for _,v,_ in fi_s],y=[k for k,_,_ in fi_s],orientation="h",
            marker_color=[CM[t] for _,_,t in fi_s],opacity=0.88,
            text=[f"{v:.4f}" for _,v,_ in fi_s],textposition="outside"))
        fig.update_layout(title="Global Feature Importance (Mean |SHAP|)",xaxis_title="Mean |SHAP value|",
            paper_bgcolor="#111318",plot_bgcolor="#111318",font=dict(color="#dde1ed",family="Epilogue"),
            title_font_size=12,height=430,margin=dict(l=10,r=80,t=40,b=40))
        st.plotly_chart(fig,use_container_width=True)
        st.markdown('<div style="font-size:0.7rem;color:#6b7491;margin-top:-0.5rem">🔴 Race &nbsp; 🔵 Geography &nbsp; ⚪ Financial</div>', unsafe_allow_html=True)

    with cp:
        st.markdown('<div class="sec-label">Live Approval Predictor</div>', unsafe_allow_html=True)
        dti_v=st.slider("DTI (%)",10,70,38); ltv_v=st.slider("LTV (%)",50,105,85)
        inc_v=st.slider("Income ($k)",30,400,120); loan_v=st.slider("Loan Amount ($k)",100,1500,450)
        race_v=st.selectbox("Race/Ethnicity",["White","Black or African American","Asian","Hispanic or Latino"])
        lt_v=st.selectbox("Loan Type",["1 - Conventional","2 - FHA","3 - VA","4 - USDA"])
        if st.button("Predict ↗", use_container_width=True):
            lt_num=int(lt_v[0]); lti_v=(loan_v*1000)/(inc_v*1000) if inc_v>0 else 3.0
            fv={"loan_amount":loan_v*1000,"income":inc_v,"dti_numeric":dti_v,"loan_to_value_ratio":ltv_v,
                "lti":lti_v,"loan_term":360,"loan_type":lt_num,"lien_status":1,
                "tract_minority_population_percent":30,"tract_to_msa_income_percentage":100,
                "race_Asian":1 if race_v=="Asian" else 0,
                "race_Black or African American":1 if race_v=="Black or African American" else 0,
                "race_Hispanic or Latino":1 if race_v=="Hispanic or Latino" else 0}
            Xp=pd.DataFrame([fv])
            for c in model_features:
                if c not in Xp.columns: Xp[c]=0
            Xp=Xp[model_features]
            pa=model.predict_proba(Xp)[0][1]; pd_=1-pa
            col="#1ec9a0" if pa>0.85 else "#f5a623" if pa>0.65 else "#e05555"
            st.markdown(f'<div style="text-align:center;padding:1.5rem;background:#111318;border:1px solid #1e2130;border-radius:12px;margin-top:1rem"><div style="font-family:DM Mono,monospace;font-size:.65rem;color:#6b7491;text-transform:uppercase;letter-spacing:.1em">Approval Probability</div><div style="font-size:2.8rem;font-weight:700;color:{col};font-family:DM Mono,monospace;line-height:1.2;margin:.5rem 0">{pa:.1%}</div><div style="font-size:.75rem;color:#6b7491">Denial probability: {pd_:.1%}</div></div>',unsafe_allow_html=True)


# ── Page: AI Analyst ──────────────────────────────────────────────────────────
elif page == "💬  AI Analyst":
    st.markdown('<div class="sec-label">AI Fair Lending Analyst — Powered by Gemini</div>', unsafe_allow_html=True)

    if gemini is None:
        st.warning("Gemini API key not configured. Add GEMINI_API_KEY to Streamlit secrets.")
    else:
        st.markdown('<div class="finding info">Ask any question about the HMDA findings, methodology, regulatory implications, or specific lenders. Full context of all Phase 1–4 results is loaded.</div>', unsafe_allow_html=True)

        SUGG=["What does the Black odds ratio of 0.665 mean legally?",
              "Which lender has the worst disparity and why does it matter?",
              "Explain Moran's I = 0.267 in plain English",
              "What would the CFPB do with these findings?",
              "Why does the gap persist even at low DTI?",
              "What is the significance of race ranking #2 in SHAP?"]
        st.markdown("**Suggested questions:**")
        cols=st.columns(3)
        for i,q in enumerate(SUGG):
            with cols[i%3]:
                if st.button(q,key=f"s{i}",use_container_width=True):
                    st.session_state["pq"]=q

        user_q=st.text_input("Your question:",value=st.session_state.get("pq",""),
            placeholder="e.g. Why is Citibank's disparity ratio so high?",key="ai_in")
        if "pq" in st.session_state: del st.session_state["pq"]

        if st.button("Ask the analyst", type="primary"):
            if user_q.strip():
                with st.spinner("Analyzing..."):
                    try:
                        resp=gemini.generate_content(f"{SYSTEM_CTX}\n\nUser question: {user_q}")
                        ans=resp.text
                        st.markdown(f'<div class="user-label">You</div><div class="user-bubble">{user_q}</div><div class="ai-label">AI Analyst</div><div class="ai-bubble">{ans}</div>',unsafe_allow_html=True)
                        if "ch" not in st.session_state: st.session_state["ch"]=[]
                        st.session_state["ch"].insert(0,{"q":user_q,"a":ans})
                    except Exception as e: st.error(f"Gemini error: {e}")

        if st.session_state.get("ch") and len(st.session_state["ch"])>1:
            st.markdown('<div class="sec-label" style="margin-top:1.5rem">Previous questions</div>', unsafe_allow_html=True)
            for item in st.session_state["ch"][1:]:
                st.markdown(f'<div class="user-label">You</div><div class="user-bubble">{item["q"]}</div><div class="ai-label">AI Analyst</div><div class="ai-bubble">{item["a"]}</div>',unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div style="font-size:.68rem;color:#3d4560;text-align:center;font-family:DM Mono,monospace;padding-bottom:1rem">HMDA Fair Lending Analysis · Washington DC MSA · 2023 FFIEC Data · For academic and research purposes only · Not legal or regulatory advice</div>',unsafe_allow_html=True)
