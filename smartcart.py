import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import numpy as np

st.set_page_config(page_title="SmartCart Customer Segmentation", layout="wide")
st.title("🛒 SmartCart — Customer Segmentation")

# ── Load Data ────────────────────────────────────────────────────────────────

df = pd.read_csv("smartcart_customers.csv")

st.subheader("Raw Data (first 5 rows)")
st.dataframe(df.head())
st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

# ── Handle Missing Values ─────────────────────────────────────────────────────

df["Income"] = df["Income"].fillna(df["Income"].median())

# ── Feature Engineering ───────────────────────────────────────────────────────

df["Age"] = 2026 - df["Year_Birth"]

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
reference_date = df["Dt_Customer"].max()
df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days

df["Total_Spending"] = (
    df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
    + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
)

df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

df["Education"] = df["Education"].replace({
    "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate", "PhD": "Postgraduate"
})

df["Living_With"] = df["Marital_Status"].replace({
    "Married": "Partner", "Together": "Partner",
    "Single": "Alone", "Divorced": "Alone",
    "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
})

# ── Drop Columns ──────────────────────────────────────────────────────────────

cols_to_drop = (
    ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer"]
    + ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
)
df_cleaned = df.drop(columns=cols_to_drop)

# ── Outlier Removal ───────────────────────────────────────────────────────────

st.subheader("Outlier Removal")
before = len(df_cleaned)
df_cleaned = df_cleaned[df_cleaned["Age"] < 90]
df_cleaned = df_cleaned[df_cleaned["Income"] < 600_000]
after = len(df_cleaned)
st.write(f"Rows before: **{before}** → Rows after removing outliers: **{after}**")

# ── Pairplot ──────────────────────────────────────────────────────────────────

st.subheader("Pairplot (selected features)")
pair_cols = ["Income", "Recency", "Response", "Age", "Total_Spending", "Total_Children"]
fig = sns.pairplot(df_cleaned[pair_cols]).fig
st.pyplot(fig)
plt.close()

# ── Correlation Heatmap ───────────────────────────────────────────────────────

st.subheader("Correlation Heatmap")
corr = df_cleaned.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(corr, annot=True, annot_kws={"size": 6}, cmap="coolwarm", ax=ax)
st.pyplot(fig)
plt.close()

# ── Encoding ──────────────────────────────────────────────────────────────────

ohe = OneHotEncoder()
cat_cols = ["Education", "Living_With"]
enc_cols = ohe.fit_transform(df_cleaned[cat_cols])
enc_df = pd.DataFrame(
    enc_cols.toarray(),
    columns=ohe.get_feature_names_out(cat_cols),
    index=df_cleaned.index
)
df_encoded = pd.concat([df_cleaned.drop(columns=cat_cols), enc_df], axis=1)

# ── Scaling ───────────────────────────────────────────────────────────────────

scaler = StandardScaler()
X = df_encoded.copy()
feature_cols = X.columns.tolist()  # save column names for predictor
X_scaled = scaler.fit_transform(X)

# ── PCA ───────────────────────────────────────────────────────────────────────

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

st.subheader("PCA — Explained Variance Ratio")
ev = pca.explained_variance_ratio_
st.write(f"PC1: **{ev[0]:.2%}** | PC2: **{ev[1]:.2%}** | PC3: **{ev[2]:.2%}**")

st.subheader("3D PCA Projection")
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.5)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_zlabel("PCA3")
ax.set_title("3D Projection")
st.pyplot(fig)
plt.close()

# ── Elbow Method ──────────────────────────────────────────────────────────────

st.subheader("Elbow Method — Optimal K")
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit_predict(X_pca)
    wcss.append(km.inertia_)

knee = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
optimal_k = knee.elbow
st.write(f"Optimal K (elbow): **{optimal_k}**")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, 11), wcss, marker="o")
ax.set_xlabel("K")
ax.set_ylabel("WCSS")
ax.set_title("Elbow Curve")
st.pyplot(fig)
plt.close()

# ── Silhouette Score ──────────────────────────────────────────────────────────

st.subheader("Silhouette Scores")
scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_pca)
    scores.append(silhouette_score(X_pca, labels))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(2, 11), scores, marker="o", color="green")
ax.set_xlabel("K")
ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Scores by K")
st.pyplot(fig)
plt.close()

# ── Combined WCSS + Silhouette Plot ──────────────────────────────────────────

st.subheader("WCSS vs Silhouette Score")
k_range = range(2, 11)
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(k_range, wcss[1:], marker="o", color="blue", label="WCSS")
ax1.set_xlabel("K")
ax1.set_ylabel("WCSS", color="blue")
ax2 = ax1.twinx()
ax2.plot(k_range, scores, marker="x", color="red", linestyle="--", label="Silhouette")
ax2.set_ylabel("Silhouette Score", color="red")
ax1.set_title("WCSS & Silhouette Score vs K")
st.pyplot(fig)
plt.close()

# ── KMeans Clustering ─────────────────────────────────────────────────────────

st.subheader("KMeans Clustering (K=4)")
kmeans = KMeans(n_clusters=4, random_state=42)
labels_kmeans = kmeans.fit_predict(X_pca)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels_kmeans, cmap="tab10", alpha=0.6)
ax.set_title("KMeans Clusters (3D PCA)")
st.pyplot(fig)
plt.close()

# ── Agglomerative Clustering ──────────────────────────────────────────────────

st.subheader("Agglomerative Clustering (K=4)")
agg_clf = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels_agg = agg_clf.fit_predict(X_pca)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels_agg, cmap="tab10", alpha=0.6)
ax.set_title("Agglomerative Clusters (3D PCA)")
st.pyplot(fig)
plt.close()

# ── Cluster Characterization ──────────────────────────────────────────────────

X["cluster"] = labels_agg
pal = ["red", "blue", "orange", "green"]

st.subheader("Cluster Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="cluster", data=X, hue="cluster", palette=pal, ax=ax, legend=False)
ax.set_title("Number of Customers per Cluster")
st.pyplot(fig)
plt.close()

st.subheader("Income vs Total Spending by Cluster")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=X["Total_Spending"], y=X["Income"], hue=X["cluster"], palette=pal, ax=ax)
ax.set_title("Income vs Spending")
st.pyplot(fig)
plt.close()

st.subheader("Cluster Summary (mean values)")
cluster_summary = X.groupby("cluster").mean()
st.dataframe(cluster_summary.style.background_gradient(cmap="Blues"))

# ── Customer Predictor ────────────────────────────────────────────────────────

st.subheader("🔮 Predict Customer Segment")
st.write("Enter a new customer's details to find which segment they belong to:")

cluster_names = {
    0: "💰 High Income, High Spender",
    1: "👨‍👩‍👧 Family Oriented, Low Spender",
    2: "🧑 Young, Low Income, Low Spender",
    3: "⭐ Medium Income, Medium Spender"
}

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Income ($)", min_value=0, max_value=200000, value=50000)

with col2:
    spending = st.number_input("Total Spending ($)", min_value=0, max_value=3000, value=500)
    recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=100, value=30)

with col3:
    children = st.number_input("Total Children", min_value=0, max_value=5, value=0)
    tenure = st.number_input("Customer Tenure (days)", min_value=0, max_value=3000, value=500)

education = st.selectbox("Education", ["Undergraduate", "Graduate", "Postgraduate"])
living_with = st.selectbox("Living With", ["Alone", "Partner"])

if st.button("🔍 Predict Cluster"):

    # Build a zero-filled row matching training columns exactly
    sample = pd.DataFrame(
        np.zeros((1, len(feature_cols))),
        columns=feature_cols
    )

    # Fill in known numeric values
    known = {
        "Income": income,
        "Recency": recency,
        "Age": age,
        "Customer_Tenure_Days": tenure,
        "Total_Spending": spending,
        "Total_Children": children,
    }
    for col, val in known.items():
        if col in sample.columns:
            sample[col] = val

    # Fill one-hot encoded education
    edu_col = f"Education_{education}"
    if edu_col in sample.columns:
        sample[edu_col] = 1

    # Fill one-hot encoded living_with
    liv_col = f"Living_With_{living_with}"
    if liv_col in sample.columns:
        sample[liv_col] = 1

    # Scale and PCA transform
    sample_scaled = scaler.transform(sample)
    sample_pca = pca.transform(sample_scaled)

    # Predict
    predicted = kmeans.predict(sample_pca)[0]
    label = cluster_names.get(predicted, f"Cluster {predicted}")

    st.success(f"✅ This customer belongs to: **{label}** (Cluster {predicted})")

    # Show average profile of predicted cluster
    st.write("**Average profile of this cluster:**")
    st.dataframe(cluster_summary.loc[[predicted]].style.background_gradient(cmap="Greens"))