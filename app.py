import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Global Dev Clustering", layout="wide")

st.title("🌍 Global Development Clustering")

# Load dataset
df = pd.read_excel("P659_World_development_dataset.xlsx")

# Load models
model = joblib.load("kmeans_model.joblib")
scaler = joblib.load("scaler.joblib")
pca = joblib.load("pca.joblib")

# Select country
country = st.selectbox("Select Country", df['Country'])

row = df[df['Country'] == country]

# Prepare features
features = df.drop(columns=['Country'], errors='ignore')

# Transform
scaled = scaler.transform(features)
pca_data = pca.transform(scaled)

# Predict clusters
clusters = model.predict(pca_data)
df['Cluster'] = clusters

cluster_id = df[df['Country'] == country]['Cluster'].values[0]

# Display
st.subheader(f"🌐 {country}")
st.write(f"Cluster Assigned: **{cluster_id}**")

# Metrics
cols = st.columns(3)
for i, col in enumerate(features.columns[:9]):
    with cols[i % 3]:
        st.metric(col, round(row[col].values[0], 2))

# Comparison chart
cluster_mean = df[df['Cluster'] == cluster_id].mean(numeric_only=True)

comparison = pd.DataFrame({
    "Country": row.iloc[0][features.columns],
    "Cluster Mean": cluster_mean[features.columns]
})

st.subheader("📊 Country vs Cluster Mean")
st.bar_chart(comparison)

# Cluster distribution
st.subheader("Cluster Distribution")
st.bar_chart(df['Cluster'].value_counts())
