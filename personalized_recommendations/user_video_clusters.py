import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
from cache_implementation import cache_data
import pickle


# Function to find videos with similar keywords
def find_similar_videos(keywords, tfidf_vectorizer, video_tfidf, videos_df, top_n=10):
    query_vector = tfidf_vectorizer.transform([keywords])
    cosine_sim = cosine_similarity(query_vector, video_tfidf).flatten()
    indices = cosine_sim.argsort()[-top_n:]
    return videos_df.iloc[indices]['Video'].tolist()



def generate_recommendations(user_clusters, tfidf_vectorizer_videos, video_tfidf, user_profiles_df, videos_df):
    recommendations = {}
    video_cluster_mapping = {}  # To track video to cluster mappings

    # Assume an approach to generate a unique outer cluster ID for demonstration
    outer_cluster_id_counter = 1
    outer_cluster_ids = {}  # Map a set of keywords or criteria to an outer cluster ID

    for _, cluster_row in user_clusters.iterrows():
        video_id = cluster_row['VidId']
        users_in_cluster = cluster_row['user_ID']
        
        # Dynamically generate an outer cluster ID (for demonstration purposes)
        # In practice, this should relate to the logic you use for clustering
        outer_cluster_id = f"outer_{outer_cluster_id_counter}"
        outer_cluster_id_counter += 1  # Increment for the next cluster
        
        video_keywords = videos_df.loc[videos_df['Video'] == video_id, 'Dominant_Topic_Keywords'].iloc[0]
        similar_video_indices = find_similar_videos(video_keywords, tfidf_vectorizer_videos, video_tfidf, videos_df, top_n=15)
        similar_videos = videos_df[videos_df['Video'].isin(similar_video_indices)]['Video'].tolist()

        for user_id in users_in_cluster:
            user_keywords = user_profiles_df.loc[user_profiles_df['user_ID'] == user_id, 'Dominant_Topic_Keywords'].values[0]
            user_specific_video_indices = find_similar_videos(user_keywords, tfidf_vectorizer_videos, video_tfidf, videos_df, top_n=5)
            user_specific_videos = videos_df[videos_df['Video'].isin(user_specific_video_indices)]['Video'].tolist()

            # Map videos to both an outer cluster ID and a user-specific inner cluster ID
            for video in set(user_specific_videos + similar_videos):
                video_cluster_mapping[video] = {"outer_cluster_id": outer_cluster_id, "inner_cluster_id": user_id}

            recommendations[user_id] = user_specific_videos + list(set(similar_videos) - set(user_specific_videos))[:5]
    


    
    return recommendations, video_cluster_mapping

# Function to get cluster centroids
def get_cluster_centroids(tfidf_vectorizer_videos, video_tfidf, user_profiles_df, recommendations, videos_df):
    # Outer cluster centroids using KMeans
    kmeans_outer = KMeans(n_clusters=5)  # Adjust n_clusters based on your dataset
    outer_centroids = kmeans_outer.fit(video_tfidf).cluster_centers_
    
    # Inner cluster centroids calculation
    inner_centroids = {}
    for user_id, recommended_videos in recommendations.items():
        if not recommended_videos:  # If no recommendations, skip to next user
            continue
        
        recommended_keywords = videos_df[videos_df['Video'].isin(recommended_videos)]['Dominant_Topic_Keywords'].dropna()
        if recommended_keywords.empty:  # If no keywords found, skip to next user
            continue
        
        recommended_tfidf = tfidf_vectorizer_videos.transform(recommended_keywords)
        inner_centroids[user_id] = np.mean(recommended_tfidf.toarray(), axis=0)
    
    return outer_centroids, inner_centroids

def main():
    # Load datasets
    comments_df = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\Datasets\\Youtube_Video_Dataset\\All_Comments_Final.csv')
    videos_df = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\enhanced_performance_metrics_with_topics.csv')
    user_profiles_df = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\user_profiles.csv')

    # Clean and preprocess data
    comments_df = comments_df[comments_df['VidId'] != '#NAME?']

    # Group users by shared video engagement to form outer clusters
    user_clusters = comments_df.groupby('VidId')['user_ID'].apply(set).reset_index()

    # Initialize TF-IDF Vectorizer for videos
    video_keywords = videos_df['Dominant_Topic_Keywords'].values.astype('U')
    tfidf_vectorizer_videos = TfidfVectorizer()
    video_tfidf = tfidf_vectorizer_videos.fit_transform(video_keywords)
    with open('tfidf_vectorizer.pkl', 'wb') as file:
        pickle.dump(tfidf_vectorizer_videos, file)

     # Now correctly placed: Generate recommendations after establishing clusters
    print("Generating recommendations")
    recommendations, video_cluster_mapping = generate_recommendations(user_clusters, tfidf_vectorizer_videos, video_tfidf, user_profiles_df, videos_df)
    print("Finished generating recommendations")
    print(f"Total videos mapped: {len(video_cluster_mapping)}")
    for cluster_id, videos in video_cluster_mapping.items():
        print(f"Cluster {cluster_id}: {len(videos)} videos")
    
    # Compute cluster centroids based on current clustering and recommendations
    outer_centroids, inner_centroids = get_cluster_centroids(tfidf_vectorizer_videos, video_tfidf, user_profiles_df, recommendations, videos_df)
   
    cache_data(outer_centroids, inner_centroids, video_cluster_mapping)

    
    print("Centroids and video-to-cluster mappings cached successfully.")

    # For demonstration: Display recommendations for a specified user
    specified_user_id = 'user_981'  # Example user ID
    # Retrieve the specified user's dominant keywords
    user_keywords = user_profiles_df.loc[user_profiles_df['user_ID'] == specified_user_id, 'Dominant_Topic_Keywords'].values
    if len(user_keywords) > 0:
        user_keywords = user_keywords[0]
    else:
        print(f"No keywords found for user {specified_user_id}.")
        user_keywords = ""

    # Assuming the recommendations for the specified user have been generated
    if specified_user_id in recommendations:
        recommended_video_ids = recommendations[specified_user_id]
    
        # Fetch the titles of the recommended videos
        recommended_video_titles = videos_df.loc[videos_df['Video'].isin(recommended_video_ids), ['Video', 'Video title']].drop_duplicates()
    
        # Print the desired output
        print(f"user_ID: {specified_user_id}")
        print(f"User's Dominant Keywords: {user_keywords}")
        print("Recommended Video Titles:")
        for _, row in recommended_video_titles.iterrows():
            print(f"- {row['Video title']} ({row['Video']})")
    else:
        print(f"No recommendations available for user {specified_user_id}.")

  

if __name__ == '__main__':
    main()



