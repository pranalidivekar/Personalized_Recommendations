import numpy as np
import pandas as pd
import redis
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from hierarchal_clustering import form_outer_clusters

"""
Author: Pranali Divekar

Description:
The code generates personalized video recommendations based on user input keywords by using TF-IDF vectors and centroids stored in Redis. 

Inputs:
- User keywords for generating recommendations (read from a CSV file).
- Precomputed TF-IDF vectorizer, centroids, and clusters (loaded from Redis).

Outputs:
- A list of recommended videos based on the closest matching clusters.
"""

def get_user_vector(user_keywords, tfidf_vectorizer):
    user_keywords_str = ' '.join(user_keywords)
    user_vector = tfidf_vectorizer.transform([user_keywords_str])
    return user_vector

def load_centroids_from_redis():
    """Load centroids for outer and inner clusters from Redis."""
    r = redis.Redis(host='localhost', port=6379, db=0)
    outer_keys = r.keys('centroid_outer_*')
    inner_keys = r.keys('centroid_inner_*')
    centroids = {'outer': {}, 'inner': {}}
    
    for key in outer_keys:
        cluster_id = key.decode('utf-8').split('_')[-1]
        centroid = pickle.loads(r.get(key))
        centroids['outer'][cluster_id] = centroid
        
    
        
    for key in inner_keys:
        key_parts = key.decode('utf-8').split('_')
        outer_cluster_id, user_id = key_parts[2], key_parts[4]
        centroid = pickle.loads(r.get(key))
        centroids['inner'].setdefault(outer_cluster_id, {})[user_id] = centroid
        
        
        
    return centroids

def find_close_clusters_for_diverse_recommendations(user_vector, centroids, threshold=0.5):
    """
    Finds close matches within both inner and outer clusters, prioritizing inner clusters,
    but falling back to multiple outer clusters if no inner match exceeds the threshold.
    """
    close_matches = []

    # First, try to find close matches in inner clusters
    for outer_cluster_id, inner_clusters in centroids['inner'].items():
        for user_id, centroid in inner_clusters.items():
            similarity = cosine_similarity(user_vector, centroid.reshape(1, -1)).flatten()[0]
            if similarity >= threshold:
                close_matches.append({'type': 'inner', 'id': (outer_cluster_id, user_id), 'similarity': similarity})

    # If no sufficient inner cluster matches are found, look in outer clusters
    if not close_matches:
        for cluster_id, centroid in centroids['outer'].items():
            similarity = cosine_similarity(user_vector, centroid.reshape(1, -1)).flatten()[0]
            if similarity >= threshold:
                close_matches.append({'type': 'outer', 'id': cluster_id, 'similarity': similarity})

    # Sort matches by similarity if needed and return
    close_matches.sort(key=lambda x: x['similarity'], reverse=True)
    return close_matches

def recommend_videos_based_on_clusters(closest_cluster_ids_dict, clusters):
    """
    Recommend videos based on the closest matching clusters identified.
    Prioritizes inner cluster matches if available.
    """
    recommended_videos = []
    for match in closest_cluster_ids_dict:
        cluster_type = match['type']
                
        # Handling outer cluster matches
        if cluster_type == 'outer':
            outer_cluster_id = match['id']
            # Fetch videos directly if it's an outer cluster match
            if outer_cluster_id in clusters:
                videos = list(clusters[outer_cluster_id]['videos'])
                recommended_videos.extend(videos)
            else:
                print(f"Outer Cluster ID {outer_cluster_id} not found in clusters.")
        
        
        # Handling inner cluster matches
        elif cluster_type == 'inner':
            outer_cluster_id, user_id_suffix  = match['id']
            user_id = f"user_{user_id_suffix}" 

            if outer_cluster_id in clusters and user_id in clusters[outer_cluster_id]['inner_clusters']:
                videos = clusters[outer_cluster_id]['inner_clusters'][user_id]
                recommended_videos.extend(videos)
            
     
    final_recommendations = list(set(recommended_videos))
    return final_recommendations[:10]

def read_keywords_from_csv(csv_path):
    """Reads keywords from a CSV file."""
    df = pd.read_csv(csv_path)
    keywords = df['Comments Keywords'].explode().tolist()
    return keywords


def main():
    r = redis.Redis(host='localhost', port=6379, db=0)
    videos_df = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\enhanced_performance_metrics_with_topics.csv', usecols=['Video', 'Video title'])
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    centroids = load_centroids_from_redis()
    print("Loaded centroids")
    clusters_data = r.get('clusters_mapping')
    if clusters_data:
        clusters = pickle.loads(clusters_data)
    else:
        print("Cluster mappings not found in cache.")
    print("Loaded clusters")
    csv_path = 'C:\\Users\\prana\\Desktop\\Capstone\\YT_comments_keywords.csv'
    keywords = read_keywords_from_csv(csv_path)
    all_recommendations = []

    for keyword in keywords:
        print("Keyword: ",keyword)
        user_vector = get_user_vector([keyword], tfidf_vectorizer)
        closest_cluster_ids_dict = find_close_clusters_for_diverse_recommendations(user_vector, centroids)
        print(f"Generated {len(closest_cluster_ids_dict)} closest clusters.")
        recommendations = recommend_videos_based_on_clusters(closest_cluster_ids_dict, clusters)
        print(recommendations)
        all_recommendations.extend(recommendations)
    
    unique_recommendations = list(set(all_recommendations))
    recommendations_with_metadata = [(video_id, videos_df[videos_df['Video'] == video_id]['Video title'].iloc[0]) for video_id in unique_recommendations if not videos_df[videos_df['Video'] == video_id].empty]

    for video_id, title in recommendations_with_metadata:
        print(f"Video ID: {video_id}, Title: {title}")

if __name__ == "__main__":
    main()
    
    
    

    

