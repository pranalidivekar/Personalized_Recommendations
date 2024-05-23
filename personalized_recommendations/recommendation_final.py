import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import redis
import json
import time

# Initialize Redis connection
redis_conn = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def fetch_data_from_redis(key):
    """Fetch data stored in Redis under the specified key."""
    cached_data = redis_conn.lrange(key, 0, -1)
    return [json.loads(item) for item in cached_data] if cached_data else []

def update_video_profiles_csv(new_video_entry, csv_path='C:\\Users\\prana\\Desktop\\Capstone\\enhanced_performance_metrics_with_topics.csv'):
    """Append a new video entry to the video profiles CSV."""
    new_entry_df = pd.DataFrame([new_video_entry])
    new_entry_df.to_csv(csv_path, mode='a', header=False, index=False)
    print("New video entry added to CSV.")

def load_user_profiles():
    """Load user profiles with dominant keywords"""
    user_profiles = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\user_profiles.csv')

    """Load user cluster mappings (inner and outer clusters)"""
    cluster_mappings = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\keywords_overlap_users_videos.csv')

    """Merge the two DataFrames on user ID"""
    combined_profiles = pd.merge(user_profiles, cluster_mappings, on='user_ID', how='left')
    
    return combined_profiles
    

def load_video_profiles(cache_key, fallback_csv_path):
    """Load video profiles from the cache or fallback to CSV if cache miss."""
    video_profiles = fetch_data_from_redis(cache_key)
    if not video_profiles:  # Cache miss, load from CSV
        print(f"Cache miss for {cache_key}. Loading from CSV.")
        video_profiles = pd.read_csv(fallback_csv_path)
    else:  # Cache hit, convert list of dicts to DataFrame
        print(f"Loaded {cache_key} from cache.")
        video_profiles = pd.DataFrame(video_profiles)
    return video_profiles


def recommend_videos(user_id, combined_user_profiles_df, cache_key='popular_videos', fallback_csv_path='C:\\Users\\prana\\Desktop\\Capstone\\enhanced_performance_metrics_with_topics.csv'):
    start_time = time.time()
    
    # Load video profiles from the cache or fallback to CSV if cache miss
    video_profiles_df = load_video_profiles(cache_key, fallback_csv_path)

    all_video_keywords = video_profiles_df['Dominant_Topic_Keywords'].tolist()
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_video_keywords)  # Learn vocabulary and IDF from all video keywords

    user_info = combined_user_profiles_df[combined_user_profiles_df['user_ID'] == user_id].iloc[0]
    user_keywords = user_info['Dominant_Topic_Keywords'].split(', ')
    user_vec = vectorizer.transform([' '.join(user_keywords)])
    video_vecs = vectorizer.transform(all_video_keywords)
    
    # Calculate cosine similarity between user keywords and video keywords
    similarities = cosine_similarity(user_vec, video_vecs).flatten()
    video_profiles_df['similarity'] = similarities
    
    # Prioritize videos based on similarity and popularity factor
    recommended_videos = video_profiles_df.sort_values(by=['similarity', 'Popularity Factor'], ascending=False)
    

    # Check for trending videos
    trending_videos = fetch_data_from_redis('trending_videos')
    trending_video_ids = [video['External Video ID'] for video in trending_videos]
    recommended_videos['Trending'] = recommended_videos['Video'].isin(trending_video_ids)
    
    # Prioritize trending videos
    recommended_videos = recommended_videos.sort_values(by=['Trending', 'similarity', 'Popularity Factor'], ascending=[False, False, False])
     
    end_time = time.time()
    latency = end_time - start_time
    print(f"Latency: {latency} seconds")
    return recommended_videos[['Video', 'Video title', 'Popularity Factor', 'Trending']].head(10)

def main():
    user_id = 'user_0'  # Example user ID
    user_profiles_df = load_user_profiles()
    cache_key = 'popular_videos'  # Cache key for popular videos
    fallback_csv_path = 'C:\\Users\\prana\\Desktop\\Capstone\\enhanced_performance_metrics_with_topics.csv'  # Fallback CSV path for popular videos
    recommendations = recommend_videos(user_id, user_profiles_df, cache_key, fallback_csv_path)
    print("Top Recommendations:")
    print(recommendations)

if __name__ == '__main__':
    main()
