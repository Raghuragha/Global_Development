# 🌍 Global Development Clustering App

## 📌 Project Overview

This project is an **Unsupervised Machine Learning application** that clusters countries based on global development indicators such as GDP, life expectancy, CO₂ emissions, and more.

The project includes:

* Model building using **KMeans clustering**
* Model evaluation using **Silhouette Score**
* Dimensionality reduction using **PCA**
* Interactive **Streamlit web application**

---

## 🚀 Features

* 📊 Automatic clustering of countries
* 📈 Optimal cluster selection using Silhouette Score
* 🔍 Country-level insights
* 📉 Comparison with cluster averages
* 🌐 Interactive Streamlit dashboard

---

## 🧠 Machine Learning Workflow

### 1. Data Preprocessing

* Handle missing values
* Feature scaling using **StandardScaler**

### 2. Dimensionality Reduction

* Applied **PCA (Principal Component Analysis)**
* Retained 95% variance

### 3. Clustering

* Used **KMeans algorithm**
* Optimal K selected using Silhouette Score

### 4. Model Evaluation

* Metric: **Silhouette Score**
* Good clustering expected between **0.4 – 0.7**

---

## 📂 Project Structure

```
├── final_merged_clustering_streamlit.ipynb
├── app.py
├── kmeans_model.joblib
├── scaler.joblib
├── pca.joblib
├── P659_World_development_dataset.xlsx
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📊 Streamlit App Features

* 🌍 Select any country
* 📌 View cluster assignment
* 📊 Key metrics display
* 📈 Compare country vs cluster mean
* 📉 Cluster distribution visualization

---

## 📸 UI Preview

The UI includes:

* Sidebar navigation
* Country selection dropdown
* Metrics cards
* Comparison charts

---

## 📦 Dependencies

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* streamlit
* joblib
* openpyxl

---

## 💾 Model Files

The following files are generated after training:

* `kmeans_model.joblib` → Clustering model
* `scaler.joblib` → Feature scaler
* `pca.joblib` → PCA transformer

---

## 🎯 Future Improvements

* Add **interactive Plotly charts**
* Improve UI with **custom CSS**
* Add **EDA & Feature Analysis tabs**
* Deploy on **Streamlit Cloud**

---

## 👨‍💻 Author

Raghavendra N
QA Engineer | Python | Automation | ML Enthusiast

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
