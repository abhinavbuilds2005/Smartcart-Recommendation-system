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

@st.cache_resource
def load_churn_model():
    with open('churn_model.pkl', 'rb') as f:
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

# Recompute explicitly from K-Means to ensure labels match! 
df_temp = df_cleaned.copy()
df_temp['Cluster'] = labels_kmeans
cluster_summary = df_temp.groupby('Cluster').mean(numeric_only=True).round(2)

SEGMENT_NAMES = {
    0: "Budget Families",
    1: "Moderate Spenders",
    2: "Premium Customers",
    3: "Budget Singles"
}

SEGMENT_DESCRIPTIONS = {
    0: "Lower income, budget-conscious spenders who typically live with a partner.",
    1: "Middle-income earners with moderate spending. Typically older families.",
    2: "High-income, high-spending premium customers. They have the fewest children.",
    3: "Lower income, budget-conscious spenders who live alone."
}

st.title("🛒 SmartCart Customer Segmentation & Intelligence")
st.markdown("Discover customer insights, predict segments dynamically using unsupervised Machine Learning, and make data-driven decisions with our **Predictive Churn Engine** and **Business Recommendation System**.")

# Project Impact Section (Sidebar)
with st.sidebar:
    st.header("🚀 Project Impact")
    st.markdown("""
    **Built an AI-Powered Customer Intelligence System using Machine Learning.**

    🔹 Segmented customers into distinct groups using K-Means clustering
    🔹 Developed a churn prediction model to identify at-risk customers
    🔹 Designed a recommendation engine to suggest personalized marketing strategies
    🔹 Created an interactive dashboard using Streamlit for real-time insights
    🔹 Improved business decision-making by combining segmentation + prediction

    💡 *This system helps businesses increase revenue, reduce churn, and target customers effectively.*
    
    ---
    🔗 **Live Demo:** [https://smartcart-ai-intell.streamlit.app](https://smartcart-ai-intell.streamlit.app)
    """)

tab1, tab2 = st.tabs(["📊 Dashboard Insights", "🔮 Predict Segment & Churn"])

# --- TAB 1: Dashboard Insights ---
with tab1:
    st.header("Cluster Overview")
    
    unique_clusters = np.unique(labels_kmeans)
    cols = st.columns(len(unique_clusters))
    for i, col in zip(unique_clusters, cols):
        cluster_size = (labels_kmeans == i).sum()
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{SEGMENT_NAMES.get(i, f'Segment {i}')}</h4>
                <p style="font-size: 24px; font-weight: bold; color: #ff7b72;">{cluster_size}</p>
                <p style="margin-bottom: 0px;">Customers</p>
                <p style="font-size: 12px; color: #888; margin-top: 0px;">Segment {i}</p>
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
    handles, labels = scatter.legend_elements()
    unique_labels = sorted(np.unique(labels_kmeans))
    legend_labels = [SEGMENT_NAMES.get(i, f"Segment {i}") for i in unique_labels]
    legend1 = ax.legend(handles, legend_labels, title="Segments", facecolor='#1f2428', edgecolor='white', labelcolor='white')
    plt.setp(legend1.get_title(), color='white')
    ax.add_artist(legend1)
    
    st.pyplot(fig)
    
    st.write("---")
    st.subheader("Marketing & Purchase Channels by Segment")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**Average Total Spending**")
        fig_spend, ax_spend = plt.subplots(figsize=(6, 4))
        fig_spend.patch.set_facecolor('#0d1117')
        ax_spend.set_facecolor('#0d1117')
        
        spending_data = df_temp.groupby('Cluster')['Total_Spending'].mean().reset_index()
        colors = ['#ff7b72', '#79c0ff', '#d2a8ff', '#ffa657']
        # Handle cases where we have fewer or more than 4 clusters gracefully
        palette = colors[:len(spending_data)] if len(spending_data) <= 4 else 'viridis'
        sns.barplot(data=spending_data, x='Cluster', y='Total_Spending', palette=palette, ax=ax_spend)
        
        ax_spend.set_xlabel('Segment', color='white')
        ax_spend.set_ylabel('Avg Total Spending', color='white')
        ax_spend.tick_params(colors='white')
        for spine in ax_spend.spines.values():
            spine.set_color('white')
        sns.despine(ax=ax_spend, left=False, bottom=False, top=True, right=True)
        ax_spend.set_xticklabels([SEGMENT_NAMES.get(int(i), f"Seg {i}") for i in spending_data['Cluster']], rotation=45, color='white')
        
        st.pyplot(fig_spend)

    with col_chart2:
        st.markdown("**Purchasing Channels Utilization**")
        fig_chan, ax_chan = plt.subplots(figsize=(6, 4))
        fig_chan.patch.set_facecolor('#0d1117')
        ax_chan.set_facecolor('#0d1117')
        
        channels = ['NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases']
        # Check if columns exist
        channels = [c for c in channels if c in df_temp.columns]
        
        if channels:
            channel_data = df_temp.groupby('Cluster')[channels].mean().reset_index()
            channel_data_melt = channel_data.melt(id_vars='Cluster', var_name='Channel', value_name='Avg Purchases')
            channel_data_melt['Channel'] = channel_data_melt['Channel'].str.replace('Num', '').str.replace('Purchases', '')
            
            sns.barplot(data=channel_data_melt, x='Cluster', y='Avg Purchases', hue='Channel', ax=ax_chan)
            
            ax_chan.set_xlabel('Segment', color='white')
            ax_chan.set_ylabel('Avg Purchases', color='white')
            ax_chan.tick_params(colors='white')
            for spine in ax_chan.spines.values():
                spine.set_color('white')
            sns.despine(ax=ax_chan, left=False, bottom=False, top=True, right=True)
            ax_chan.set_xticklabels([SEGMENT_NAMES.get(int(i), f"Seg {i}") for i in channel_data['Cluster']], rotation=45, color='white')
            
            legend = ax_chan.legend(title='Channel', facecolor='#1f2428', edgecolor='white', labelcolor='white')
            plt.setp(legend.get_title(), color='white')
            
            st.pyplot(fig_chan)
        else:
            st.info("Purchase channel data not available.")
    
    st.write("---")
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
        
        # Predict Segment
        prediction = kmeans.predict(input_pca)[0]
        
        # Predict Churn
        churn_model_data = load_churn_model()
        churn_model = churn_model_data['model']
        churn_features = churn_model_data['features']
        
        churn_input = pd.DataFrame([{
            'Age': age, 'Income': income, 'Recency': recency, 'NumDealsPurchases': num_deals,
            'NumWebPurchases': num_web, 'NumCatalogPurchases': num_catalog, 
            'NumStorePurchases': num_store, 'NumWebVisitsMonth': num_web_visits,
            'Customer_Tenure_Days': customer_tenure, 'Total_Spending': total_spending,
            'Total_Children': total_children
        }])
        
        churn_input = churn_input[churn_features]
        churn_prob = churn_model.predict_proba(churn_input)[0][1]
        churn_risk = "HIGH" if churn_prob > 0.50 else "LOW"
        
        # Determine business logic based on segment
        if prediction == 2:
            segment_icon = "💎"
            segment_tag = "Premium Customer"
            spending_tag = "High Spending"
            business_insight = "These customers generate high revenue and are strongly engaged with the brand."
            rec_strategy = "Target with premium offers and exclusive deals"
            rec_campaign = "VIP Email Marketing"
            rec_offer = "Early Access to Luxury Products"
            rec_priority = "High"
            rec_actions = ["Promote luxury products", "Give early access deals"]
        elif prediction == 0:
            segment_icon = "🛒"
            segment_tag = "Budget Family"
            spending_tag = "Budget Conscious"
            business_insight = "These customers are highly price-sensitive and respond well to discounts."
            rec_strategy = "Provide discounts and bundle offers"
            rec_campaign = "Discount/Promo Newsletters"
            rec_offer = "20% Discount / Buy 1 Get 1"
            rec_priority = "Medium"
            rec_actions = ["Send bundle offers", "Provide family bulk discounts"]
        elif prediction == 1:
            segment_icon = "👨‍👩‍👧‍👦"
            segment_tag = "Moderate Spender Family"
            spending_tag = "Moderate Spending"
            business_insight = "These customers form the stable core of the business with reliable, steady purchases."
            rec_strategy = "Encourage upselling and cross-selling"
            rec_campaign = "Targeted Product Recommendations"
            rec_offer = "Loyalty Points Multiplier"
            rec_priority = "Medium"
            rec_actions = ["Recommend complementary items", "Promote loyalty rewards"]
        else: # 3
            segment_icon = "👤"
            segment_tag = "Budget Single"
            spending_tag = "Low Spending"
            business_insight = "These customers are budget-conscious individuals who buy strictly what they need."
            rec_strategy = "Nudge towards more frequent purchases with low-barrier offers"
            rec_campaign = "Flash Sales Campaigns"
            rec_offer = "10% Off Next Purchase"
            rec_priority = "Low"
            rec_actions = ["Push limited-time flash sales", "Recommend lower-tier affordable items"]

        # Churn specific actions
        if churn_risk == "HIGH":
            churn_actions = ["Send immediate retention offer", "Provide loyalty rewards"]
            churn_color = "#ff7b72"
        else:
            churn_actions = ["Maintain current engagement", "Upsell standard products"]
            churn_color = "#3fb950"

        if prediction == 2:
            st.balloons()
            
        st.markdown(f"## {segment_icon} {segment_tag}")
        st.markdown(f"**📈 {spending_tag} | ⚠️ Churn Risk: <span style='color:{churn_color}'>{churn_risk}</span>**", unsafe_allow_html=True)
        
        st.markdown(f"**📌 Business Insight:**\n{business_insight}")
        
        st.markdown("---")
        col_rec, col_churn = st.columns(2)
        
        with col_rec:
            st.markdown("### 🎯 Recommended Strategy")
            st.markdown(f"""
            - **Strategy:** {rec_strategy}
            - **Campaign:** {rec_campaign}
            - **Offer:** {rec_offer}
            - **Priority:** {rec_priority}
            
            **📌 Recommended Action:**
            """)
            for action in rec_actions:
                st.markdown(f"- {action}")
                
        with col_churn:
            st.markdown("### 🚨 Churn Management")
            st.markdown(f"""
            - **Risk Level:** <span style='color:{churn_color}; font-weight:bold;'>{churn_risk}</span>
            - **Churn Probability:** {churn_prob:.1%}
            
            **📌 Action:**
            """, unsafe_allow_html=True)
            for action in churn_actions:
                st.markdown(f"- {action}")
                
        st.markdown("---")
        st.write("#### 📊 Detailed Averages for this Segment")
        st.dataframe(cluster_summary.loc[[prediction]].style.background_gradient(cmap='Blues'))
