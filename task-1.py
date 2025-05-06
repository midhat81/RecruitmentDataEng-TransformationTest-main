import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset (replace 'your_data.csv' with your dataset path)
df = pd.read_csv('Customer_Data.csv')

# 1. Data Loading and Understanding
print("Missing values:\n", df.isnull().sum())
print("Duplicate records:\n", df.duplicated().sum())

# Basic Statistics
print("Basic statistics:\n", df.describe())

# 2. Exploratory Data Analysis (EDA)
# Visualize Avg_Credit_Limit distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Avg_Credit_Limit'], bins=30, kde=True)
plt.title('Avg Credit Limit Distribution')
plt.xlabel('Avg Credit Limit')
plt.ylabel('Frequency')
plt.show()

# Visualize Total visits to bank vs Total visits online
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Total_visits_bank'], y=df['Total_visits_online'])
plt.title('Total Visits Bank vs Total Visits Online')
plt.xlabel('Total Visits to Bank')
plt.ylabel('Total Visits Online')
plt.show()

# 3. Feature Engineering (For Frequency and Monetary)
# We are using visits and calls as proxies for Frequency
df['Frequency'] = df['Total_visits_bank'] + df['Total_visits_online'] + df['Total_calls_made']

# Monetary is represented by Avg_Credit_Limit
df['Monetary'] = df['Avg_Credit_Limit']

# Create a new DataFrame for RFM features
rfm_df = df[['Customer Key', 'Frequency', 'Monetary']]

# 4. Customer Segmentation (Clustering)
# Standardizing RFM data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df[['Frequency', 'Monetary']])

# Use KMeans for clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

# 5. Cluster Analysis and Interpretation
# Add cluster centroids to the RFM dataframe for interpretation
centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=['Frequency', 'Monetary'])

# Show average values per cluster
cluster_summary = rfm_df.groupby('Cluster').agg({
    'Frequency': 'mean',
    'Monetary': 'mean'
}).reset_index()

print("Cluster Summary:\n", cluster_summary)

# Visualize the clusters in 2D using PCA
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=rfm_pca[:, 0], y=rfm_pca[:, 1], hue=rfm_df['Cluster'], palette='Set2', s=100, alpha=0.7)
plt.title('Customer Segments')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.show()

# 6. Business Strategy Recommendations
for cluster_id, cluster_data in cluster_summary.iterrows():
    if cluster_data['Frequency'] > 20 and cluster_data['Monetary'] > 5000:
        print(f"Cluster {cluster_id} - High Value: Focus on loyalty programs and personalized offers.")
    elif cluster_data['Frequency'] < 5 and cluster_data['Monetary'] < 1000:
        print(f"Cluster {cluster_id} - At Risk: Send re-engagement offers.")
    elif cluster_data['Frequency'] > 10 and cluster_data['Monetary'] < 2000:
        print(f"Cluster {cluster_id} - Frequent Shoppers: Offer bundled discounts to increase purchase value.")
    else:
        print(f"Cluster {cluster_id} - New Customers: Implement onboarding strategies to retain them.")
