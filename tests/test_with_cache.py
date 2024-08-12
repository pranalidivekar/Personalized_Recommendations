import time
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from hierarchal_clustering import form_outer_clusters

"""
Author: Pranali Divekar

Description:
The code generates personalized video recommendations based on user input keywords. 
It uses precomputed TF-IDF vectors and centroids to find the closest matching video clusters 
and recommends videos from those clusters. 
The recommendations are based on the similarity between the user's keywords and the centroids of the clusters.

Inputs:
- User input keywords for generating recommendations.
- Precomputed TF-IDF vectorizer, centroids, and clusters (loaded from cache).

Outputs:
- A list of recommended videos based on the closest matching clusters.
"""

def get_user_vector(user_keywords, tfidf_vectorizer):
    """Convert user keywords into a TF-IDF vector."""
    user_keywords_str = ' '.join(user_keywords)
    user_vector = tfidf_vectorizer.transform([user_keywords_str])
    return user_vector

'''def load_centroids_from_redis():
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
        
        
        
    return centroids'''

def find_close_clusters_for_diverse_recommendations(user_vector, centroids, threshold=0.5):
    """
    Find clusters that are close to the user's keyword vector based on cosine similarity.
    Searches inner clusters first, then outer clusters if no close matches are found.
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
            user_id = user_id_suffix
            
            # Fetch videos from the specific user's inner cluster
            if outer_cluster_id in clusters and user_id in clusters[outer_cluster_id]['inner_clusters']:
                videos = clusters[outer_cluster_id]['inner_clusters'][user_id]
                recommended_videos.extend(videos)
           

  
    final_recommendations = list(set(recommended_videos))
    return final_recommendations[:10]

def read_keywords_from_csv(csv_path):
    """Reads keywords from a CSV file."""
    df = pd.read_csv(csv_path)
    # Flatten the DataFrame into a list of keywords
    keywords = df['Comments Keywords'].explode().tolist()
    return keywords


def main():
    start_time = time.time()
    #r = redis.Redis(host='localhost', port=6379, db=0)
    videos_df = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\enhanced_performance_metrics_with_topics.csv', usecols=['Video', 'Video title'])
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    #centroids = load_centroids_from_redis()
    # Load centroids
    with open('centroids.pkl', 'rb') as file:
        centroids = pickle.load(file)
   

    '''clusters_data = r.get('clusters_mapping')
    if clusters_data:
        clusters = pickle.loads(clusters_data)
    else:
        print("Cluster mappings not found in cache.")'''
    
    # Load clusters
    with open('clusters.pkl', 'rb') as file:
        clusters = pickle.load(file)
    user_keywords = ['data', 'scientist']

    user_vector = get_user_vector(user_keywords, tfidf_vectorizer)
    closest_cluster_ids_dict = find_close_clusters_for_diverse_recommendations(user_vector, centroids)
    recommendations = recommend_videos_based_on_clusters(closest_cluster_ids_dict, clusters)

    recommendations_with_metadata = [(video_id, videos_df[videos_df['Video'] == video_id]['Video title'].iloc[0]) for video_id in recommendations if not videos_df[videos_df['Video'] == video_id].empty]
    
    
    #display output
    for video_id, title in recommendations_with_metadata:
        print(f"Link: {video_id}, Title: {title}")
    print("Recommendation generation time:", time.time() - start_time, "seconds")


if __name__ == "__main__":
    main()