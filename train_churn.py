import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('smartcart_customers.csv', sep=',')
df.dropna(inplace=True)

# ✅ Feature engineering FIRST
current_year = datetime.datetime.now().year
df['Age'] = current_year - df['Year_Birth']
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format="%d-%m-%Y", errors='coerce')
df['Customer_Tenure_Days'] = (pd.to_datetime('2024-01-01') - df['Dt_Customer']).dt.days
df['Customer_Tenure_Days'] = df['Customer_Tenure_Days'].fillna(500)
df['Total_Spending'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
df['Total_Children'] = df['Kidhome'] + df['Teenhome']

# ✅ Churn label
np.random.seed(42)
df['Churn'] = ((df['Recency'] > 60) | (df['Complain'] == 1)).astype(int)

# ✅ Recency and Complain removed to fix data leakage
features = ['Age', 'Income', 'NumDealsPurchases', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
            'Customer_Tenure_Days', 'Total_Spending', 'Total_Children']

X = df[features]
y = df['Churn']

# ✅ Fix fillna
X = X.fillna(X.median())

# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ class_weight balanced for imbalanced data
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# ✅ Print real accuracy metrics
print(classification_report(y_test, model.predict(X_test)))

# ✅ Save model
with open('churn_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': features}, f)

print("Churn model trained and saved as churn_model.pkl")
