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
# PREPARE FEATURES (FIXED)
# ===============================
features = df[feature_columns]

# Handle missing values
features = features.fillna(features.mean())

# Transform
scaled = scaler.transform(features)
pca_data = pca.transform(scaled)

# Predict
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
    with cols[i % 3]:
        st.metric(
            label=col,
            value=round(row[col].values[0], 2)
        )

# ===============================
# COMPARISON GRAPH
# ===============================
st.subheader("📈 Country vs Cluster Mean")

cluster_mean = df[df['Cluster'] == cluster_id][feature_columns].mean()

comparison = pd.DataFrame({
    "Country": row.iloc[0][feature_columns],
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
