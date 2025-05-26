import streamlit as st 
import pandas as pd 
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Title
st.set_page_config(page_title="Credit Card Fraud Trends", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    df['Amount_scaled'] = StandardScaler().fit_transform(df[['Amount']])
    df['Time_hours'] = df['Time'] / 3600
    return df

df = load_data()

# Sidebar
st.sidebar.header("Filter Transactions")
amount_range = st.sidebar.slider("Transaction Amount", 0.0, float(df['Amount'].max()), (0.0, 1000.0))
df_filtered = df[(df['Amount'] >= amount_range[0]) & (df['Amount'] <= amount_range[1])]

# Visual 1: Fraud Over Time
st.subheader("💢Fraudulent Transactions Over Time (Hours)💢")
fraud_times = df_filtered[df_filtered['Class'] == 1].groupby(df_filtered['Time_hours'].round())['Class'].count()
st.line_chart(fraud_times)

# Visual 2: Amount Distribution
st.subheader("💰 Transaction Amount Distribution")
fig1 = px.histogram(df_filtered, x='Amount', color=df_filtered['Class'].map({0: 'Legit', 1: 'Fraud'}),
                    nbins=100, barmode='overlay', labels={'color': 'Transaction Type'})
st.plotly_chart(fig1, use_container_width=True)

# PCA + Clustering
st.subheader("🧠 Fraud Segmentation (PCA + KMeans)")
features = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled']
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[features])
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(pca_data)
df['PCA1'], df['PCA2'], df['Cluster'] = pca_data[:, 0], pca_data[:, 1], clusters

fig2 = px.scatter(df.sample(3000), x='PCA1', y='PCA2', color='Cluster',
                  title="Transaction Clusters (Risk Segmentation)")
st.plotly_chart(fig2, use_container_width=True)

# Fraud Prediction Section
st.subheader("📎 Fraud Prediction (Random Forest)")

with st.expander("Try Predicting Fraud from Features"):
    input_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
    user_input = [st.number_input(col, value=0.0) for col in input_cols]
    predict_df = pd.DataFrame([user_input], columns=input_cols)

    # Prepare training data
    X = df[input_cols]
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    if st.button("Predict"):
        prediction = model.predict(predict_df)[0]
        prob = model.predict_proba(predict_df)[0][1]
        st.success(f"Prediction: {'Fraud' if prediction == 1 else 'Legit'} (Probability: {prob:.2%})")

# Footer
st.markdown("---")
st.caption("Built using Streamlit & Plotly | Dataset: Kaggle Credit Card Fraud")
