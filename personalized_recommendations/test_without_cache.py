
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from collections import defaultdict

def group_users_by_video_interaction(comments_df):
    """
    Groups users based on the videos they have commented on.
    """
    user_groups = defaultdict(set)
    for _, row in comments_df.iterrows():
        user_groups[row['VidId']].add(row['user_ID'])
    return user_groups

def form_outer_clusters(user_groups, videos_df, user_profiles_df):
    """
    Forms outer clusters and refines them based on user interests.
    """
    videos_df['Dominant_Topic_Keywords'] = videos_df['Dominant_Topic_Keywords'].fillna('')
    # Initialize TF-IDF Vectorizer based on all videos' dominant keywords
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(videos_df['Dominant_Topic_Keywords'].values)

    clusters = {}
    outer_cluster_counter = 1
    for video_id, users in user_groups.items():
        filtered_df = videos_df[videos_df['Video'] == video_id]
        if not filtered_df.empty:
            target_keywords = filtered_df['Dominant_Topic_Keywords'].iloc[0]
            similar_videos = find_similar_videos(target_keywords, videos_df, tfidf_vectorizer)
            #print(f"Found {len(similar_videos)} similar videos for video ID {video_id}")
            for user_id in users:
                user_keywords = user_profiles_df[user_profiles_df['user_ID'] == user_id]['Dominant_Topic_Keywords'].iloc[0]
                refined_videos = refine_videos_for_user(user_keywords, similar_videos, videos_df, tfidf_vectorizer)
                #print(f"Found {len(refined_videos)} refined videos for user ID {user_id} in cluster {video_id}")
                # Assume each user forms an inner cluster within the outer cluster of the original video
                cluster_name = str(outer_cluster_counter)
                if cluster_name not in clusters:
                    clusters[cluster_name] = {'inner_clusters': {}, 'videos': set(similar_videos)}
                    outer_cluster_counter += 1
                    
                clusters[cluster_name]['inner_clusters'][user_id] = refined_videos
                clusters[cluster_name]['videos'].update(refined_videos)
        else:
            print(f"No matching video found for Video ID: {video_id}")
    
    return clusters

def find_similar_videos(target_keywords, videos_df, tfidf_vectorizer, top_n=10):
  target_vector = tfidf_vectorizer.transform([target_keywords])
  all_vectors = tfidf_vectorizer.transform(videos_df['Dominant_Topic_Keywords'].values)
  cosine_similarities = cosine_similarity(target_vector, all_vectors).flatten()
  similar_indices = cosine_similarities.argsort()[-top_n:]
  return videos_df.iloc[similar_indices]['Video'].tolist()

def refine_videos_for_user(user_keywords, candidate_videos, videos_df, tfidf_vectorizer):
    user_vector = tfidf_vectorizer.transform([user_keywords])
    candidate_vectors = tfidf_vectorizer.transform(videos_df[videos_df['Video'].isin(candidate_videos)]['Dominant_Topic_Keywords'].values)
    cosine_similarities = cosine_similarity(user_vector, candidate_vectors).flatten()
    # Assuming the top half are considered more relevant
    top_indices = cosine_similarities.argsort()[-len(candidate_videos)//2:]
    refined_videos = videos_df.iloc[top_indices]['Video'].tolist()
    return refined_videos

def calculate_centroids(videos_df, clusters):
    
    tfidf_vectorizer = TfidfVectorizer()
    video_keywords = [videos_df[videos_df['Video'] == vid]['Dominant_Topic_Keywords'].iloc[0] for vid in videos_df['Video']]
    tfidf_matrix = tfidf_vectorizer.fit_transform(video_keywords)

    centroids = {'outer': {}, 'inner': {}}

    for outer_cluster_id, cluster_data in clusters.items():
        outer_videos = list(cluster_data['videos'])
        outer_vectors = tfidf_vectorizer.transform(videos_df[videos_df['Video'].isin(outer_videos)]['Dominant_Topic_Keywords'].values)
        outer_centroid = np.mean(outer_vectors.toarray(), axis=0)  # Ensure it's dense for mean calculation
        centroids['outer'][outer_cluster_id] = outer_centroid
        
        for user_id, inner_videos in cluster_data['inner_clusters'].items():
            if inner_videos:
                inner_vectors = tfidf_vectorizer.transform(videos_df[videos_df['Video'].isin(inner_videos)]['Dominant_Topic_Keywords'].values)
                inner_centroid = np.mean(inner_vectors.toarray(), axis=0)
                centroids['inner'].setdefault(outer_cluster_id, {})[user_id] = inner_centroid

    
    return centroids, tfidf_vectorizer

def get_user_vector(user_keywords, tfidf_vectorizer):
    # Assuming `user_keywords` is a list of keywords
    user_keywords_str = ' '.join(user_keywords)
    user_vector = tfidf_vectorizer.transform([user_keywords_str])
    return user_vector

def find_close_clusters_for_diverse_recommendations(user_vector, centroids):
    """
    Finds close matches within both inner and outer clusters, prioritizing inner clusters,
    but falling back to multiple outer clusters if no inner match exceeds the threshold.
    """
    close_matches = []
    threshold=0.5
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
    #print(f"Generated {len(clusters)} clusters.")
    #print(f"Generated {len(closest_cluster_ids_dict)} closest clusters.")
    recommended_videos = []
    for match in closest_cluster_ids_dict:
        cluster_type = match['type'] #inner or outer
                
        # Handling outer cluster matches
        if cluster_type == 'outer':
            outer_cluster_id = match['id']
            # Fetch videos directly if it's an outer cluster match
            if outer_cluster_id in clusters:
                videos = list(clusters[outer_cluster_id]['videos'])
                recommended_videos.extend(videos)
               
            
        
        
        # Handling inner cluster matches
        elif cluster_type == 'inner':
            #print(f"Debug: match['id'] for inner = {match['id']}")  # Diagnostic print

            outer_cluster_id, user_id_suffix  = match['id']
            user_id = user_id_suffix
            
            # Fetch videos from the specific user's inner cluster
            if outer_cluster_id in clusters and user_id in clusters[outer_cluster_id]['inner_clusters']:
                videos = clusters[outer_cluster_id]['inner_clusters'][user_id]
                recommended_videos.extend(videos)
              
     
    # Deduplicate while maintaining order
    final_recommendations = list(set(recommended_videos))
    
    # Here, you can apply further logic to prioritize or limit the number of recommendations
    return final_recommendations[:10]


# Test function to demonstrate usage
def test_recommendation_system(user_keywords):
    start_time = time.time()

    comments_df = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\Datasets\\Youtube_Video_Dataset\\All_Comments_Final.csv')
    videos_df = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\enhanced_performance_metrics_with_topics.csv')
    user_profiles_df = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\user_profiles.csv')

    # Group users by video interaction
    user_groups = group_users_by_video_interaction(comments_df)

    # Directly forming clusters and calculating centroids
    clusters = form_outer_clusters(user_groups, videos_df, user_profiles_df)
    centroids, tfidf_vectorizer = calculate_centroids(videos_df, clusters)
    #print("Time taken to calculate clusters and centroids:", time.time() - start_time, "seconds")
    # Getting recommendations for the user
    user_vector = get_user_vector(user_keywords, tfidf_vectorizer)
    closest_cluster = find_close_clusters_for_diverse_recommendations(user_vector, centroids)
  
    recommendations = recommend_videos_based_on_clusters(closest_cluster, clusters)
    print(recommendations)
    print("Time taken:", time.time() - start_time, "seconds")

if __name__ == "__main__":
    user_keywords = ['learn', 'would', 'start', 'project', 'reviewing', 'episode', 'portfolio', 'changed', 'resume', 'github']
    test_recommendation_system(user_keywords)
