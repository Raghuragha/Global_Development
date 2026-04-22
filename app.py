import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(layout="wide")

st.title("🌍 Global Dev Clustering")
st.subheader("Unsupervised ML Project")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    return pd.read_excel("P659_World_development_dataset.xlsx")

df = load_data()

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    model = joblib.load("kmeans_model.joblib")
    scaler = joblib.load("scaler.joblib")
    pca = joblib.load("pca.joblib")
    return model, scaler, pca

model, scaler, pca = load_models()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

# ===============================
# COUNTRY SELECT
# ===============================
country = st.selectbox("Select country to inspect", df['Country'])
row = df[df['Country'] == country]

# ===============================
# ✅ EXACT TRAINING PIPELINE (FIX)
# ===============================

# Step 1: Drop same column
features = df.drop(columns=['Country'], errors='ignore')

# Step 2: Convert to numeric
features = features.apply(pd.to_numeric, errors='coerce')

# Step 3: Fill missing values
features = features.fillna(features.mean())

# Step 4: Convert to numpy (IMPORTANT)
features_array = features.values

# ===============================
# TRANSFORM + PREDICT
# ===============================
scaled = scaler.transform(features_array)
pca_data = pca.transform(scaled)

clusters = model.predict(pca_data)
df['Cluster'] = clusters

cluster_id = df[df['Country'] == country]['Cluster'].values[0]

# ===============================
# DISPLAY METRICS
# ===============================
st.markdown(f"### 🌐 {country}")
st.write(f"Cluster assigned: **{cluster_id}**")

col1, col2, col3 = st.columns(3)

display_cols = features.columns[:9]

for i, col in enumerate(display_cols):
    value = row[col].values[0] if col in row else None

    try:
        value = float(value)
        value = round(value, 2)
    except:
        value = "N/A"

    with [col1, col2, col3][i % 3]:
        st.metric(col, value)

# ===============================
# COMPARISON GRAPH
# ===============================
cluster_mean = df[df['Cluster'] == cluster_id][display_cols].mean(numeric_only=True)

comparison = pd.DataFrame({
    "Country": pd.to_numeric(row[display_cols].iloc[0], errors='coerce'),
    "Cluster Mean": cluster_mean
})

st.subheader("📊 Country vs Cluster Mean")
st.bar_chart(comparison)

# ===============================
# CLUSTER DISTRIBUTION
# ===============================
st.subheader("Cluster Distribution")
st.bar_chart(df['Cluster'].value_counts())
