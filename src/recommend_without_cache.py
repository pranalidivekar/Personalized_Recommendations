from collections import defaultdict
import pickle
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd

"""
Author: Pranali Divekar

Description:
The code provides a video recommendation system based on clustering YouTube video data without using caching. 

Inputs:
- YouTube video data with dominant topic keywords in CSV format.
- User keywords as input for generating video recommendations.

Outputs:
- A list of recommended videos based on the closest matching cluster.
"""

videos_df = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\enhanced_performance_metrics_with_topics.csv')

def calculate_clusters(videos_df):
    videos_df['Dominant_Topic_Keywords'] = videos_df['Dominant_Topic_Keywords'].fillna('')
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(videos_df['Dominant_Topic_Keywords'])
    
    # Use KMeans for clustering; assuming 2 clusters for simplicity
    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(tfidf_matrix)
    
    # Assign cluster labels to videos
    videos_df['Cluster'] = kmeans.labels_
    
    # Calculate centroids
    centroids = kmeans.cluster_centers_
    
    return videos_df, centroids

def recommend_videos(user_keywords, videos_df, centroids, tfidf_vectorizer):
    # Vectorize user keywords
    user_keywords_str = ', '.join(user_keywords) if isinstance(user_keywords, list) else user_keywords
    
    user_vector = tfidf_vectorizer.transform([user_keywords_str])
    
    # Calculate similarity with centroids
    similarity = cosine_similarity(user_vector, centroids)
    
    # Identify closest cluster
    closest_cluster = np.argmax(similarity)
    
    # Recommend videos from the closest cluster
    recommended_videos = videos_df[videos_df['Cluster'] == closest_cluster]['Video'].tolist()
    
    return recommended_videos

def main(user_keywords):
    start_time = time.time()
    
    # Calculate clusters and centroids
    videos_df_with_clusters, centroids = calculate_clusters(videos_df)
    
    # Reuse the TF-IDF Vectorizer for user keywords
    tfidf_vectorizer = TfidfVectorizer().fit(videos_df['Dominant_Topic_Keywords'])
    
    # Generate recommendations
    recommendations = recommend_videos(user_keywords, videos_df_with_clusters, centroids, tfidf_vectorizer)
    
    end_time = time.time()
    print(f"Recommendations: {recommendations}")
    print(f"Time taken for recommendations (without cache): {end_time - start_time:.2f} seconds")

# Example user keywords input
user_keywords = ['learn', 'would', 'start', 'project', 'reviewing', 'episode', 'portfolio', 'changed', 'resume', 'github']

main(user_keywords)
