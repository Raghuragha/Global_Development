import streamlit as st
import pandas as pd
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Global Development Clustering", layout="wide")

st.title("🌍 Global Development Clustering Dashboard")

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
# COUNTRY SELECT
# ===============================
country = st.selectbox("🌐 Select Country", df["Country"])
row = df[df["Country"] == country]

# ===============================
# 🔥 EXACT FEATURE MATCH (CRITICAL)
# ===============================

# Get expected number of features
expected_features = scaler.n_features_in_

# Drop Country
features = df.drop(columns=["Country"], errors="ignore")

# Convert to numeric
features = features.apply(pd.to_numeric, errors='coerce')

# Fill missing
features = features.fillna(features.mean())

# Convert to numpy
features_array = features.values

# 🔥 MATCH FEATURE COUNT
current_features = features_array.shape[1]

if current_features > expected_features:
    features_array = features_array[:, :expected_features]

elif current_features < expected_features:
    import numpy as np
    padding = np.zeros((features_array.shape[0], expected_features - current_features))
    features_array = np.hstack((features_array, padding))

# ===============================
# TRANSFORM + PREDICT
# ===============================
scaled = scaler.transform(features_array)
pca_data = pca.transform(scaled)

clusters = model.predict(pca_data)
df["Cluster"] = clusters

cluster_id = df[df["Country"] == country]["Cluster"].values[0]

# ===============================
# DISPLAY
# ===============================
st.subheader(f"📍 {country}")
st.write(f"Cluster Assigned: **{cluster_id}**")

# ===============================
# METRICS
# ===============================
st.subheader("📊 Key Indicators")

cols = st.columns(3)
display_cols = df.columns[1:10]

for i, col in enumerate(display_cols):
    value = row[col].values[0] if col in row else None

    try:
        value = float(value)
        value = round(value, 2)
    except:
        value = "N/A"

    with cols[i % 3]:
        st.metric(col, value)

# ===============================
# CHARTS
# ===============================
st.subheader("📈 Country vs Cluster Mean")

cluster_mean = df[df["Cluster"] == cluster_id][display_cols].mean(numeric_only=True)

comparison = pd.DataFrame({
    "Country": pd.to_numeric(row[display_cols].iloc[0], errors='coerce'),
    "Cluster Mean": cluster_mean
})

st.bar_chart(comparison)

st.subheader("📉 Cluster Distribution")
st.bar_chart(df["Cluster"].value_counts())
