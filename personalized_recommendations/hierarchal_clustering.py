from collections import defaultdict
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd

def group_users_by_video_interaction(comments_df):
    # Groups users based on the videos they have commented on.
    
    user_groups = defaultdict(set)
    for _, row in comments_df.iterrows():
        user_groups[row['VidId']].add(row['user_ID'])
    return user_groups

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
 

def form_outer_clusters(user_groups, videos_df, user_profiles_df):
    videos_df['Dominant_Topic_Keywords'] = videos_df['Dominant_Topic_Keywords'].fillna('')
    # vectorizer
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


def calculate_and_save_centroids_to_redis(videos_df, clusters):
    #r = redis.Redis(host='localhost', port=6379, db=0)  # Connect to Redis
    tfidf_vectorizer = TfidfVectorizer()
    video_keywords = [videos_df[videos_df['Video'] == vid]['Dominant_Topic_Keywords'].iloc[0] for vid in videos_df['Video']]
    tfidf_matrix = tfidf_vectorizer.fit_transform(video_keywords)
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)


    centroids = {'outer': {}, 'inner': {}}

    for outer_cluster_id, cluster_data in clusters.items():
        outer_videos = list(cluster_data['videos'])
        outer_vectors = tfidf_vectorizer.transform(videos_df[videos_df['Video'].isin(outer_videos)]['Dominant_Topic_Keywords'].values)
        outer_centroid = np.mean(outer_vectors.toarray(), axis=0)  # Ensure it's dense for mean calculation
        centroids['outer'][outer_cluster_id] = outer_centroid
        #r.set(f'centroid_outer_{outer_cluster_id}', pickle.dumps(outer_centroid))
        
        for user_id, inner_videos in cluster_data['inner_clusters'].items():
            if inner_videos:
                inner_vectors = tfidf_vectorizer.transform(videos_df[videos_df['Video'].isin(inner_videos)]['Dominant_Topic_Keywords'].values)
                inner_centroid = np.mean(inner_vectors.toarray(), axis=0)
                centroids['inner'].setdefault(outer_cluster_id, {})[user_id] = inner_centroid
                #r.set(f'centroid_inner_{outer_cluster_id}_{user_id}', pickle.dumps(inner_centroid))

   
    
    return centroids

def print_cluster_hierarchy(clusters):
    for outer_cluster_id, data in clusters.items():
        print(f"Outer Cluster ID: {outer_cluster_id}, Videos: {list(data['videos'])}")
        for user_id, videos in data['inner_clusters'].items():
            print(f"  Inner Cluster for User {user_id}: {videos}")

def main():
    start_time = time.time()
    # Load datasets
    comments_df = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\Datasets\\Youtube_Video_Dataset\\All_Comments_Final.csv')
    videos_df = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\enhanced_performance_metrics_with_topics.csv')
    user_profiles_df = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\user_profiles.csv')
    

    # Group users by video interaction
    user_groups = group_users_by_video_interaction(comments_df)
    
    # Form outer clusters
    clusters = form_outer_clusters(user_groups, videos_df, user_profiles_df)

    # Save clusters
    with open('clusters.pkl', 'wb') as file:
        pickle.dump(clusters, file)
    
    print(f"Generated {len(clusters)} clusters.")
    #r = redis.Redis(host='localhost', port=6379, db=0)
    clusters_data = pickle.dumps(clusters)
    #r.set('clusters_mapping', clusters_data)
    print("Cluster data pushed to cache")
    # Calculate centroids for these clusters
    centroids = calculate_and_save_centroids_to_redis(videos_df, clusters)
    # Save centroids
    with open('centroids.pkl', 'wb') as file:
        pickle.dump(centroids, file)
    print("Saved clusters to cache!")
    #print_cluster_hierarchy(clusters)

    # Print execution time
    print("Execution time for clustering and centroids calculation:", time.time() - start_time, "seconds")

   
if __name__ == "__main__":
    main()
