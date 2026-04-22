import streamlit as st
import pandas as pd
import joblib

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
    feature_columns = joblib.load("features.joblib")
    return model, scaler, pca, feature_columns

df = load_data()
model, scaler, pca, feature_columns = load_models()

# ===============================
# COUNTRY SELECTION
# ===============================
country = st.selectbox("🌐 Select Country", df['Country'])
row = df[df['Country'] == country]

# ===============================
# PREPARE FEATURES (BULLETPROOF)
# ===============================

# Ensure ALL required columns exist
for col in feature_columns:
    if col not in df.columns:
        df[col] = 0  # add missing column

# Keep exact order
features = df[feature_columns].copy()

# Convert safely to numeric (DO NOT DROP columns)
for col in features.columns:
    features[col] = pd.to_numeric(features[col], errors='coerce')

# Fill missing values column-wise
for col in features.columns:
    features[col].fillna(features[col].mean(), inplace=True)

# FINAL safety (no NaN allowed)
features = features.fillna(0)

# ===============================
# TRANSFORM & PREDICT
# ===============================
scaled = scaler.transform(features)
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

for i, col in enumerate(feature_columns[:9]):
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

cluster_mean = df[df['Cluster'] == cluster_id][feature_columns].mean(numeric_only=True)

country_values = row[feature_columns].iloc[0]

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
