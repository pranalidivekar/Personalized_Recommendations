import numpy as np
import redis
import json

"""
Author: Pranali Divekar

Description:
This script handles the caching and retrieval of centroids and video-to-cluster mappings in Redis. 
It provides functions to cache data, list all keys in the Redis cache, and retrieve cached centroids 
and video cluster mappings. The centroids are cached as JSON-serialized arrays, and the video cluster 
mappings are stored as JSON-encoded dictionaries.

Inputs:
- Centroid data (outer and inner) and video-to-cluster mappings.

Outputs:
- Cached data in Redis and functions to retrieve this data for later use.
"""

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)


def cache_data(outer_centroids, inner_centroids, video_cluster_mapping):
    r.set("outer_centroids", json.dumps(outer_centroids.tolist()))
    r.set("inner_centroids", json.dumps({k: v.tolist() for k, v in inner_centroids.items()}))
    r.set("video_cluster_mapping", json.dumps(video_cluster_mapping))

    print("Message from cache: Centroids and mappings cached successfully.")

def list_all_keys():
    """List all keys in the Redis cache."""
    keys = r.keys('*')
    print("Keys in Redis:")
    for key in keys:
        print(key.decode('utf-8'))
   

def get_cached_centroids():
    """Retrieve cached centroids from Redis."""
    outer_centroids = json.loads(r.get("outer_centroids") or "[]")
    inner_centroids = json.loads(r.get("inner_centroids") or "{}")
    outer_centroids = np.array(outer_centroids)
    inner_centroids = {k: np.array(v) for k, v in inner_centroids.items()}
    
    return outer_centroids, inner_centroids

def get_cached_video_cluster_mapping():
    """Retrieve cached video-to-cluster mappings from Redis."""
    video_cluster_mapping = json.loads(r.get("video_cluster_mapping") or "{}")
    return video_cluster_mapping


if __name__ == '__main__':
    list_all_keys()


