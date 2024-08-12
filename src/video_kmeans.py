from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

"""
Author: Pranali Divekar

Description:
The code performs K-means clustering on video profiles to segment the videos into clusters based on their popularity and geographic distribution. 
It uses the 'Elbow Method' to determine the optimal k (for clusters) and assigns cluster labels to the dataset for further analysis.

Inputs:
- CSV file containing performance metrics of YouTube videos (generated via video_profiles.py).

Outputs:
- A CSV file with added cluster labels for each video.
- A plot showing the Elbow Method for determining the optimal number of clusters.
"""

performance_metrics = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\performance_metrics.csv')

features_to_scale = ['Popularity Factor']
features_to_encode = ['Country Code']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler())])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, features_to_scale),
        ('cat', categorical_transformer, features_to_encode)])

X = preprocessor.fit_transform(performance_metrics)

#kmeans
wcss = []  # Within-cluster sum of square
for i in range(1, 11):  # Test k values from 1 to 10
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the results onto a line graph to observe the 'elbow'
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within cluster sum of squares
plt.xticks(np.arange(1, 11, 1))
plt.grid(True)
plt.show()

optimal_k = 3
kmeans_optimal = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans_optimal.fit(X)

# Add the cluster labels to your performance_metrics DataFrame
performance_metrics['Cluster'] = kmeans_optimal.labels_
print(performance_metrics.head())