import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from gensim import corpora, models
from gensim.utils import simple_preprocess

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Function to preprocess text data
def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def identify_trending_videos(video_data):
    # Calculate net likes
    video_data['Net Likes'] = video_data['Video Likes Added'] - (video_data['Video Dislikes Added'] + video_data['Video Likes Removed'])
    
    # Filter videos based on trending criteria
    trending_criteria = (video_data['Views'] >= 1000) & (video_data['Net Likes'] >= 100)
    return video_data[trending_criteria]

def main():
    video_performance_over_time = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\Datasets\\Youtube_Video_Dataset\\Video_Performance_Over_Time.csv', encoding='ISO-8859-1')
    video_performance_over_time['Video Title'] = video_performance_over_time['Video Title'].fillna('')
    video_performance_over_time['processed_titles'] = video_performance_over_time['Video Title'].apply(preprocess_text)

    # Identify trending videos
    trending_videos = identify_trending_videos(video_performance_over_time)
    print("Printing trending videos")
    print(trending_videos.head())
    print("===============================")

    texts = [simple_preprocess(title) for title in trending_videos['processed_titles'] if title.strip()]
    if not texts:
        print("No titles with content after preprocessing.")
        return
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=1, no_above=0.5, keep_n=100000)
    corpus = [dictionary.doc2bow(text) for text in texts if text]
    if not corpus:
        print("Corpus is empty after filtering.")
        return

    # Apply LDA
    lda_model = models.LdaMulticore(corpus=corpus, num_topics=10, id2word=dictionary, passes=10, workers=2)
    
    # Extract topic keywords from the LDA model for trending videos
    topic_keywords = {i: [word for word, prob in lda_model.show_topic(i)] for i in range(lda_model.num_topics)}
    trending_videos['Dominant_Topic_Keywords'] = [
        ', '.join(topic_keywords[sorted(lda_model[corpus[i]], key=lambda x: (x[1]), reverse=True)[0][0]])
        for i in range(len(trending_videos))
    ]

    # Save or print the trending videos with dominant keywords
    trending_videos.to_csv('trending_videos.csv', index=False)
    print(trending_videos[['External Video ID','Video Title', 'Dominant_Topic_Keywords']].head())

if __name__ == '__main__':
    main()
