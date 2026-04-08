import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="SmartCart Segmentation", page_icon="🛒", layout="wide")

# Custom CSS for rich aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #58a6ff;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        transform: scale(1.05);
    }
    .metric-card {
        background: linear-gradient(145deg, #1f2428, #24292e);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 5px 5px 10px #080a0f, -5px -5px 10px #121825;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model_data = load_model()
kmeans = model_data['kmeans']
scaler = model_data['scaler']
pca = model_data['pca']
ohe = model_data['ohe']
feature_cols = model_data['feature_cols']
X_pca = model_data['X_pca']
labels_kmeans = model_data['labels_kmeans']
df_cleaned = model_data['df_cleaned']
cluster_summary = model_data['cluster_summary']

st.title("🛒 SmartCart Customer Segmentation")
st.markdown("Discover customer insights and predict segments dynamically using unsupervised Machine Learning.")

tab1, tab2 = st.tabs(["📊 Dashboard Insights", "🔮 Predict Segment"])

# --- TAB 1: Dashboard Insights ---
with tab1:
    st.header("Cluster Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    for i, col in enumerate([col1, col2, col3, col4]):
        cluster_size = (labels_kmeans == i).sum()
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Segment {i}</h3>
                <p style="font-size: 24px; font-weight: bold; color: #ff7b72;">{cluster_size}</p>
                <p>Customers</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.write("---")
    
    st.subheader("Income vs Spending by Segment")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    scatter = ax.scatter(df_cleaned['Total_Spending'], df_cleaned['Income'], 
                         c=labels_kmeans, cmap='viridis', alpha=0.7, edgecolors='w', linewidth=0.5)
    
    ax.set_xlabel('Total Spending', color='white', fontsize=12)
    ax.set_ylabel('Income', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Custom legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Segments", facecolor='#1f2428', edgecolor='white', labelcolor='white')
    plt.setp(legend1.get_title(), color='white')
    ax.add_artist(legend1)
    
    st.pyplot(fig)
    
    st.subheader("Segment Characteristics Summary")
    st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'))

# --- TAB 2: Predict Segment ---
with tab2:
    st.header("Enter Customer Details")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            income = st.number_input("Income ($)", min_value=0, max_value=200000, value=50000)
            total_spending = st.number_input("Total Spending", min_value=0, value=500)
            customer_tenure = st.number_input("Customer Tenure (Days)", min_value=0, value=500)
            recency = st.number_input("Recency (Days since last purchase)", min_value=0, value=10)
            
        with col2:
            num_deals = st.number_input("Num Deals Purchases", min_value=0, value=2)
            num_web = st.number_input("Num Web Purchases", min_value=0, value=5)
            num_catalog = st.number_input("Num Catalog Purchases", min_value=0, value=2)
            num_store = st.number_input("Num Store Purchases", min_value=0, value=5)
            num_web_visits = st.number_input("Num Web Visits / Month", min_value=0, value=5)
            
        with col3:
            total_children = st.number_input("Total Children", min_value=0, value=1)
            complain = st.selectbox("Complain in last 2 years?", [0, 1])
            response = st.selectbox("Accepted recent campaign?", [0, 1])
            education = st.selectbox("Education Level", ["Undergraduate", "Graduate", "Postgraduate"])
            living_with = st.selectbox("Living Situation", ["Alone", "Partner"])
            
        submit_button = st.form_submit_button("Predict Segment 🔮")

    if submit_button:
        # Preprocess input
        input_data = pd.DataFrame([{
            'Income': income, 'Recency': recency, 'NumDealsPurchases': num_deals,
            'NumWebPurchases': num_web, 'NumCatalogPurchases': num_catalog, 
            'NumStorePurchases': num_store, 'NumWebVisitsMonth': num_web_visits,
            'Complain': complain, 'Response': response, 'Age': age, 
            'Customer_Tenure_Days': customer_tenure, 'Total_Spending': total_spending,
            'Total_Children': total_children, 'Education': education, 'Living_With': living_with
        }])
        
        # One Hot Encoding
        cat_cols = ['Education', 'Living_With']
        enc_cols = ohe.transform(input_data[cat_cols])
        enc_df = pd.DataFrame(enc_cols.toarray(), columns=ohe.get_feature_names_out(cat_cols))
        
        input_cleaned = pd.concat([input_data.drop(columns=cat_cols), enc_df], axis=1)
        
        # Ensure column order matches training data
        for col in feature_cols:
            if col not in input_cleaned.columns:
                input_cleaned[col] = 0
        input_cleaned = input_cleaned[feature_cols]
        
        # Scale and PCA
        input_scaled = scaler.transform(input_cleaned)
        input_pca = pca.transform(input_scaled)
        
        # Predict
        prediction = kmeans.predict(input_pca)[0]
        
        st.success(f"### 🎉 Customer Belongs to **Segment {prediction}**!")
        
        st.balloons()
        
        # Explain briefly what this segment means based on the summary
        st.info("Segment Characteristics based on dataset average:")
        st.write(cluster_summary.iloc[prediction:prediction+1])
