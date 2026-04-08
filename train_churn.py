import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load the same dataset
df = pd.read_csv('smartcart_customers.csv', sep=',')
# Some preprocessing to match roughly
df.dropna(inplace=True)

# Define a Churn target: if Recency is high (e.g., > 60) let's assume they are churned (this is just for demo purposes)
# We can also add some randomness or dependence on other variables
np.random.seed(42)
# Churn risk increases with Recency, complain, and less spending
# We'll create a synthetic label to train the model
df['Churn'] = ((df['Recency'] > 60) | (df['Complain'] == 1)).astype(int)

# Select simple numerical features
features = ['Age', 'Income', 'Recency', 'NumDealsPurchases', 'NumWebPurchases', 
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 
            'Customer_Tenure_Days', 'Total_Spending', 'Total_Children']

# We don't have all these columns directly in the raw CSV, so let's compute them same as notebook probably
import datetime
current_year = 2024
df['Age'] = current_year - df['Year_Birth']
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format="%d-%m-%Y", errors='coerce')
df['Customer_Tenure_Days'] = (pd.to_datetime('2024-01-01') - df['Dt_Customer']).dt.days
df['Customer_Tenure_Days'].fillna(500, inplace=True)
df['Total_Spending'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
df['Total_Children'] = df['Kidhome'] + df['Teenhome']

X = df[['Age', 'Income', 'Recency', 'NumDealsPurchases', 'NumWebPurchases', 
        'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 
        'Customer_Tenure_Days', 'Total_Spending', 'Total_Children']]
y = df['Churn']

# In case there are NaNs from calculation
X.fillna(X.median(), inplace=True)

# Train a RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
with open('churn_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': X.columns.tolist()}, f)

print("Churn model trained and saved as churn_model.pkl")
