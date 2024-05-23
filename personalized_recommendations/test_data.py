import requests
import pandas as pd
from gensim.utils import simple_preprocess
from gensim import models, corpora
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Assuming stopwords and WordNetLemmatizer are already set up
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    if not tokens:
        return None 
    return tokens

def extract_keywords_lda(processed_texts, num_topics=5):
    if not processed_texts:
        return []  # Return an empty list if no texts to process
    
    dictionary = corpora.Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=1, no_above=0.5, keep_n=10000)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    if not corpus:
        return []  # Return an empty list if corpus is empty
    
    lda_model = models.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=2)
    keywords = set()
    for idx in range(num_topics):
        keywords.update(word for word, _ in lda_model.show_topic(idx))
    return list(keywords)

def process_and_save_keywords(all_comments):
    nltk.download('punkt')
    nltk.download('stopwords')
    
    processed_titles = []
    processed_comments = []

    for video_id, info in all_comments.items():
        # Process the title
        preprocessed_title = preprocess_text(info['title'])
        if preprocessed_title:  # If not None, add to the list
            processed_titles.append(preprocessed_title)
        
        # Process comments
        for comment in info['comments']:
            preprocessed_comment = preprocess_text(comment)
            if preprocessed_comment:  # If not None, add to the list
                processed_comments.append(preprocessed_comment)
    
    # Extract keywords
    title_keywords = extract_keywords_lda(processed_titles)
    comment_keywords = extract_keywords_lda(processed_comments)
    
    # Save to CSV
    pd.DataFrame({'Video Titles Keywords': title_keywords}).to_csv('YT_video_keywords.csv', index=False)
    pd.DataFrame({'Comments Keywords': comment_keywords}).to_csv('YT_comments_keywords.csv', index=False)

    print("Keywords extracted and saved to CSVs.")



def fetch_similar_videos(api_key, query, max_results=10):
    search_url = 'https://www.googleapis.com/youtube/v3/search'
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'maxResults': max_results,
        'key': api_key
    }
    
    response = requests.get(search_url, params=params)
    search_results = response.json()
    
    videos = []
    for item in search_results.get('items', []):
        video_id = item['id']['videoId']
        title = item['snippet']['title']
        videos.append({'video_id': video_id, 'title': title})
    
    return videos

def fetch_comments_for_video(api_key, video_id, max_results=5):
    comments_url = 'https://www.googleapis.com/youtube/v3/commentThreads'
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'maxResults': max_results,
        'textFormat': 'plainText',
        'key': api_key
    }
    
    response = requests.get(comments_url, params=params)
    comments_data = response.json()
    
    comments = []
    for item in comments_data.get('items', []):
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)
    
    return comments

           
if __name__ == "__main__":
    api_key = 'AIzaSyDKfjOEn5TlZxP9T8e1H8R3ZZ8C72HosCs'
    keywords = ["kaggle", "data science"]
    max_videos = 10
    max_comments = 5
    all_comments = {}
    for keyword in keywords:
        videos = fetch_similar_videos(api_key, keyword, max_results=max_videos)
        
        for video in videos:
            video_id = video['video_id']
            video_title = video['title']
            comments = fetch_comments_for_video(api_key, video_id, max_results=max_comments)
            all_comments[video_id] = {'title': video_title, 'comments': comments}  
            

    process_and_save_keywords(all_comments)