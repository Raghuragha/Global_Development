import streamlit as st
import pandas as pd
import joblib
import numpy as np

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
    return (
        joblib.load("kmeans_model.joblib"),
        joblib.load("scaler.joblib"),
        joblib.load("pca.joblib")
    )

model, scaler, pca = load_models()

# ===============================
# SELECT COUNTRY
# ===============================
country = st.selectbox("🌐 Select Country", df["Country"])
row = df[df["Country"] == country]

# ===============================
# 🔥 ROBUST FEATURE PIPELINE
# ===============================

# Step 1: Drop non-feature column
features = df.drop(columns=["Country"], errors="ignore")

# Step 2: Convert to numeric
features = features.apply(pd.to_numeric, errors="coerce")

# Step 3: Fill missing
features = features.fillna(features.mean())

# Step 4: Convert to numpy
features_array = features.values

# ===============================
# 🔥 MATCH FEATURE COUNT (CRITICAL FIX)
# ===============================
expected = scaler.n_features_in_

current = features_array.shape[1]

if current > expected:
    # Trim extra columns
    features_array = features_array[:, :expected]

elif current < expected:
    # Add missing columns as zeros
    padding = np.zeros((features_array.shape[0], expected - current))
    features_array = np.hstack((features_array, padding))

# ===============================
# TRANSFORM
# ===============================
scaled = scaler.transform(features_array)
pca_data = pca.transform(scaled)

# ===============================
# PREDICT
# ===============================
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
# COMPARISON
# ===============================
st.subheader("📈 Country vs Cluster Mean")

cluster_mean = df[df["Cluster"] == cluster_id][display_cols].mean(numeric_only=True)

comparison = pd.DataFrame({
    "Country": pd.to_numeric(row[display_cols].iloc[0], errors="coerce"),
    "Cluster Mean": cluster_mean
})

st.bar_chart(comparison)

# ===============================
# DISTRIBUTION
# ===============================
st.subheader("📉 Cluster Distribution")
st.bar_chart(df["Cluster"].value_counts())

st.markdown("---")
st.caption("KMeans + PCA + Streamlit")
