import redis
import json
import pandas as pd

"""
Author: Pranali Divekar

Description:
The code loads data to Redis

Inputs:
- CSV files containing trending videos, popular videos, and user profiles.

Outputs:
- Cached video data and user profiles in Redis.
"""

# Initialize Redis connection
redis_conn = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def load_data_to_redis(key, data):
    """Load data into Redis under the specified key. Data should be a list of dictionaries."""
    redis_conn.delete(key)
    for item in data:
        redis_conn.rpush(key, json.dumps(item))

def fetch_data_from_redis(key):
    """Fetch data stored in Redis under the specified key. Returns a list of dictionaries."""
    cached_data = redis_conn.lrange(key, 0, -1)
    return [json.loads(item) for item in cached_data] if cached_data else None

def prepare_and_cache_videos(trending_videos_path, popular_videos_path, user_profiles_path):
    """Load trending and popular video data from CSV files and cache them in Redis."""
    # Load video datasets
    trending_videos = pd.read_csv(trending_videos_path)
    popular_videos = pd.read_csv(popular_videos_path)
    user_profiles = pd.read_csv(user_profiles_path)

    # Convert DataFrames to list of dictionaries for Redis
    trending_videos_list = trending_videos.to_dict('records')
    popular_videos_list = popular_videos.to_dict('records')
    user_profiles_list = user_profiles.to_dict('records') 


    # Cache video data
    load_data_to_redis('trending_videos', trending_videos_list)
    load_data_to_redis('popular_videos', popular_videos_list)
    load_data_to_redis('user_profiles', user_profiles_list)  


    print("Videos cached successfully.")

def print_user_profile_keys_from_cache():
    user_profile_str = redis_conn.lrange('user_profiles', 0, 0) 

    if user_profile_str:
        user_profile_dict = json.loads(user_profile_str[0])
        print("Column names in user profile:", list(user_profile_dict.keys()))
    else:
        print("No user profiles found in cache.")

def main():
    trending_videos_path = 'C:\\Users\\prana\\Desktop\\Capstone\\trending_videos.csv'  
    popular_videos_path = 'C:\\Users\\prana\\Desktop\\Capstone\\popular_videos.csv'
    user_profiles_path = 'C:\\Users\\prana\\Desktop\\Capstone\\user_profiles.csv'
    prepare_and_cache_videos(trending_videos_path, popular_videos_path, user_profiles_path)
    print_user_profile_keys_from_cache()

if __name__ == '__main__':
    main()
