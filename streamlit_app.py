import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier

# Page Setup
st.set_page_config(page_title="Production ML Development Dashboard", layout="wide")
st.title("🚀 Production-Grade Global Development Analytics Workspace")
st.caption("Advanced Cluster Diagnostics, Boundary Anomaly Detection, and Explainable AI Engine")

# ==========================================
# 📂 FILE LOADING
# ==========================================
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

if not csv_files:
    st.info("💡 Please drag and drop your CSV file into the VS Code folder on the left side panel.")
    st.stop()

selected_file = csv_files[0]
df = pd.read_csv(selected_file)

# ==========================================
# 🧠 SMART COLUMN PROCESSING & DATA CLEANING
# ==========================================
country_col = "Country" if "Country" in df.columns else df.select_dtypes(include=[object]).columns[0]

# Auto-detect structural columns to calculate Per Capita metric
gdp_col = [c for c in df.columns if "gdp" in c.lower()]
pop_col = [c for c in df.columns if "pop" in c.lower() or "people" in c.lower()]

if gdp_col and pop_col:
    # Clean string punctuation fields
    df[gdp_col[0]] = pd.to_numeric(df[gdp_col[0]].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    df[pop_col[0]] = pd.to_numeric(df[pop_col[0]].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    df["GDP Per Capita"] = df[gdp_col[0]] / df[pop_col[0]]
    st.sidebar.success("✅ Calculated 'GDP Per Capita' dynamically!")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(num_cols) < 1:
    st.error("Error: Your dataset needs at least one numeric feature column.")
    st.stop()

# ==========================================
# 🎛 SIDEBAR ADVANCED CONTROLS
# ==========================================
st.sidebar.header("🎛 ML Config Engine")
k = 3

default_features = [c for c in num_cols if "capita" in c.lower() or "expectancy" in c.lower() or "fertility" in c.lower()]
if not default_features:
    default_features = num_cols[:min(3, len(num_cols))]

features = st.sidebar.multiselect(
    "📊 Clustering Features",
    num_cols,
    default=default_features
)

if not features:
    st.warning("Please select at least one indicator in the sidebar.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Outlier Handling")
outlier_threshold = st.sidebar.slider(
    "Truncate Extreme Outliers (Percentile)",
    90, 100, 100, 
    help="Lowering this to 98% drops the top 2% of extreme values to make comparisons clearer."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 UI Display Selector")
chart_choice = st.sidebar.radio(
    "Primary Workspace View:",
    ["ML Model Diagnostics & Feature Importance", "Feature Distribution & Outlier Profiles", "Interactive Country Ranking Matrix"]
)

# ==========================================
# 🧠 EXTENDED OUTLIER HANDLING & TRAINING
# ==========================================
df_clean = df.dropna(subset=features + [country_col]).copy()

# Force string cleaning on currency/commas
for col in features:
    df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")
df_clean = df_clean.dropna(subset=features).reset_index(drop=True)

# Truncate outliers based on selected sidebar percentile
if outlier_threshold < 100:
    for col in features:
        upper_limit = np.percentile(df_clean[col], outlier_threshold)
        df_clean = df_clean[df_clean[col] <= upper_limit]

X = df_clean[features].copy()

# Log transform heavily skewed indicators
X_log = X.copy()
for col in features:
    if X[col].max() > 5000: 
        X_log[col] = np.log1p(X[col])

X_scaled = (X_log - X_log.mean()) / X_log.std()

# Live Agglomerative ML Training
agg_model = AgglomerativeClustering(n_clusters=k)
df_clean["Cluster"] = agg_model.fit_predict(X_scaled)

# Explicit labeling step sorted by primary feature
sort_metric = "GDP Per Capita" if "GDP Per Capita" in df_clean.columns else features[0]
cluster_order = df_clean.groupby("Cluster")[sort_metric].mean().sort_values().index

labels = ["Under-Developed", "Developing", "Developed"]
cluster_map = dict(zip(cluster_order, labels))
df_clean["Development Level"] = df_clean["Cluster"].map(cluster_map)

# ==========================================
# 🚨 ADVANCED ANOMALY & BORDERLINE DETECTION
# ==========================================
# Use an internal random forest classifier to extract classification probabilities
rf = RandomForestClassifier(random_state=42)
rf.fit(X_scaled, df_clean["Development Level"])
probs = rf.predict_proba(X_scaled)
max_probs = np.max(probs, axis=1)

# Flag countries where the model is highly uncertain (margin of victory is tight)
df_clean["Model Confidence %"] = np.round(max_probs * 100, 1)
borderline_df = df_clean[df_clean["Model Confidence %"] < 65]

# ==========================================
# 📈 KPI METRICS PANEL
# ==========================================
sil_score = round(silhouette_score(X_scaled, df_clean["Cluster"]), 3)

m1, m2, m3, m4 = st.columns(4)
m1.metric("🌍 Total Records Evaluated", len(df_clean))
m2.metric("📈 Silhouette Score", sil_score)
m3.metric("🚨 Boundary Anomalies Found", len(borderline_df))
with m4:
    st.write("") 
    st.download_button(
        label="⬇️ Export Complete Clean Model Data",
        data=df_clean.to_csv(index=False),
        file_name="production_ml_development_report.csv",
        mime="text/csv"
    )

st.divider()

# ==========================================
# 📊 CATEGORY WORKSPACE LAYOUT
# ==========================================
st.subheader("📋 Machine Learning Classification Breakdown")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔴 Under-Developed")
    st.dataframe(df_clean[df_clean["Development Level"] == "Under-Developed"][[country_col, "Model Confidence %"]], height=180, use_container_width=True)

with col2:
    st.markdown("### 🟡 Developing")
    st.dataframe(df_clean[df_clean["Development Level"] == "Developing"][[country_col, "Model Confidence %"]], height=180, use_container_width=True)

with col3:
    st.markdown("### 🟢 Developed")
    st.dataframe(df_clean[df_clean["Development Level"] == "Developed"][[country_col, "Model Confidence %"]], height=180, use_container_width=True)

# Anomaly Warning Callout Banner
if not borderline_df.empty:
    st.warning(f"⚠️ **Borderline Category Shift Warning**: The machine learning model identifies **{len(borderline_df)}** countries transitioning directly on the boundary margins between structural levels.")
    st.dataframe(borderline_df[[country_col, "Development Level", "Model Confidence %"]].sort_values(by="Model Confidence %"), use_container_width=True)

st.divider()

# ==========================================
# 🎯 WORKSPACE VIEW CONDITIONALS
# ==========================================
if chart_choice == "ML Model Diagnostics & Feature Importance":
    st.subheader("🧠 Model Transparency & Explainable AI (XAI)")
    
    # Calculate feature importance via Random Forest
    importances = rf.feature_importances_
    feat_imp_df = pd.DataFrame({"Feature": features, "Impact Weight Score": importances}).sort_values(by="Impact Weight Score", ascending=True)
    
    diag_c1, diag_c2 = st.columns(2)
    with diag_c1:
        fig_imp = px.bar(feat_imp_df, x="Impact Weight Score", y="Feature", orientation="h", title="Which Metric Drives the Clustering Most?")
        st.plotly_chart(fig_imp, use_container_width=True)
    with diag_c2:
        fig_scatter = px.scatter(
            df_clean, x=features[0], y=features[min(1, len(features)-1)],
            color="Development Level", size="Model Confidence %", hover_name=country_col,
            title="Spatial Cluster Boundary Overlaps (Bubble Size = Classification Certainty)",
            color_discrete_map={"Developed": "#636efa", "Developing": "#fecb52", "Under-Developed": "#ef553b"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

elif chart_choice == "Feature Distribution & Outlier Profiles":
    st.subheader("📊 Distribution Spread Profile Analysis")
    box_feat = st.selectbox("Pick Metric to Analyze Structural Spread:", features)
    fig_box = px.box(
        df_clean, x="Development Level", y=box_feat, color="Development Level",
        points="all", title=f"Distribution Profile of {box_feat}",
        color_discrete_map={"Developed": "#636efa", "Developing": "#fecb52", "Under-Developed": "#ef553b"}
    )
    st.plotly_chart(fig_box, use_container_width=True)

else:
    st.subheader("📊 Performance Matrix Ranking Engine")
    bar_feat = st.selectbox("Pick Metric to Rank Countries:", features)
    top_n = st.slider("Number of Countries to Show", 5, 50, 20)
    df_sorted = df_clean.sort_values(by=bar_feat, ascending=False).head(top_n)
    
    fig_bar = px.bar(
        df_sorted, x=country_col, y=bar_feat, color="Development Level",
        title=f"Top {top_n} Countries ranked by {bar_feat}",
        color_discrete_map={"Developed": "#636efa", "Developing": "#fecb52", "Under-Developed": "#ef553b"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ==========================================
# 📍 INDIVIDUAL COUNTRY DRILLDOWN
# ==========================================
st.subheader("📍 Target Country Drilldown & Baseline Analysis")
selected_country = st.selectbox("Select a Specific Country to Examine:", sorted(df_clean[country_col].unique()))

country_row = df_clean[df_clean[country_col] == selected_country].iloc[0]
global_means = df_clean[features].mean()

st.markdown(f"📊 **Machine Learning Summary for {selected_country}:** Assigned Tier -> **{country_row['Development Level']}** (Model Confidence Level: `{country_row['Model Confidence %']}%`)")

drill_cols = st.columns(len(features))
for idx, feat in enumerate(features):
    with drill_cols[idx]:
        c_val = country_row[feat]
        g_avg = global_means[feat]
        pct_diff = ((c_val - g_avg) / g_avg) * 100 if g_avg != 0 else 0
        st.metric(
            label=f"{feat}",
            value=f"{c_val:,.2f}",
            delta=f"{pct_diff:+.1f}% vs Global Avg"
        )

        
