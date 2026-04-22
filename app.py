import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Global Development Clustering",
    layout="wide"
)

st.title("🌍 Global Development Clustering Dashboard")

# ===============================
# LOAD DATA & MODELS
# ===============================
@st.cache_data
def load_data():
    return pd.read_excel("P659_World_development_dataset.xlsx")

@st.cache_resource
def load_models():
    model = joblib.load("kmeans_model.joblib")
    scaler = joblib.load("scaler.joblib")
    pca = joblib.load("pca.joblib")
    return model, scaler, pca

df = load_data()
model, scaler, pca = load_models()

# ===============================
# COUNTRY SELECTION
# ===============================
country = st.selectbox("🌐 Select Country", df['Country'])
row = df[df['Country'] == country]

# ===============================
# PREPARE FEATURES (MATCH TRAINING)
# ===============================

# Drop non-numeric columns like during training
features = df.drop(columns=['Country'], errors='ignore')

# Keep only numeric columns
features = features.select_dtypes(include=['number'])

# Fill missing values
features = features.fillna(features.mean())

# 🔥 CRITICAL: MATCH EXACT FEATURE COUNT
expected_features = scaler.n_features_in_

if features.shape[1] > expected_features:
    features = features.iloc[:, :expected_features]

elif features.shape[1] < expected_features:
    for i in range(expected_features - features.shape[1]):
        features[f"missing_{i}"] = 0

# Convert to numpy
features_array = features.values

# ===============================
# TRANSFORM & PREDICT
# ===============================
scaled = scaler.transform(features_array)
pca_data = pca.transform(scaled)

clusters = model.predict(pca_data)
df['Cluster'] = clusters

cluster_id = df[df['Country'] == country]['Cluster'].values[0]

# ===============================
# DISPLAY INFO
# ===============================
st.subheader(f"📍 {country}")
st.write(f"Cluster Assigned: **{cluster_id}**")

# ===============================
# METRICS DISPLAY
# ===============================
st.subheader("📊 Key Indicators")

cols = st.columns(3)

display_cols = features.columns[:9]

for i, col in enumerate(display_cols):
    value = row[col].values[0] if col in row else None

    try:
        value = float(value)
        value = round(value, 2)
    except:
        value = "N/A"

    with cols[i % 3]:
        st.metric(label=col, value=value)

# ===============================
# COMPARISON GRAPH
# ===============================
st.subheader("📈 Country vs Cluster Mean")

cluster_mean = df[df['Cluster'] == cluster_id][display_cols].mean(numeric_only=True)
country_values = row[display_cols].iloc[0]

comparison = pd.DataFrame({
    "Country": pd.to_numeric(country_values, errors='coerce'),
    "Cluster Mean": cluster_mean
})

st.bar_chart(comparison)

# ===============================
# CLUSTER DISTRIBUTION
# ===============================
st.subheader("📉 Cluster Distribution")

st.bar_chart(df['Cluster'].value_counts())

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Built using KMeans Clustering | PCA | Streamlit")
