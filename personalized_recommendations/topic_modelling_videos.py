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
    # Lowercase the text
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def identify_popular_videos(video_data):
    
    # Filter videos based on popular criteria
    popularity_criteria = (video_data['Popularity Factor'] >= 100)
    return video_data[popularity_criteria]


def main():
    # Assuming 'metrics_by_video' has already been loaded as part of your video profiles preparation
    # Preprocess video titles
    metrics_by_video = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\Datasets\\Youtube_Video_Dataset\\Aggregated_Metrics_By_Video.csv', encoding='ISO-8859-1')
    # Fill NaN values in 'Video title' with an empty string before applying the preprocessing
    metrics_by_video['Video title'] = metrics_by_video['Video title'].fillna('')
    metrics_by_video['processed_titles'] = metrics_by_video['Video title'].apply(preprocess_text)


    performance_metrics = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\performance_metrics_updated.csv')

    texts = [simple_preprocess(title) for title in metrics_by_video['processed_titles']]
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Apply LDA
    lda_model = models.LdaMulticore(corpus=corpus, num_topics=10, id2word=dictionary, passes=10, workers=2)

    # Extract topic keywords from the LDA model
    topic_keywords = {i: [word for word, prob in lda_model.show_topic(i)] for i in range(lda_model.num_topics)}
    # Assign topic keywords to each video based on the dominant topic
    metrics_by_video['Dominant_Topic_Keywords'] = [', '.join(topic_keywords[sorted(lda_model[corpus[i]], key=lambda x: (x[1]), reverse=True)[0][0]]) for i in range(len(metrics_by_video))]
    # Save or print the updated video profiles
    #metrics_by_video.to_csv('updated_metrics_by_video_with_topics.csv', index=False)
    print(metrics_by_video[['Video', 'Video title', 'Dominant_Topic_Keywords']].head())
    print("=============================================================================")
    performance_metrics = pd.merge(performance_metrics, metrics_by_video[['Video', 'Dominant_Topic_Keywords']], on='Video', how='left')
    performance_metrics.to_csv('enhanced_performance_metrics_with_topics.csv', index=False)
    print(performance_metrics[['Video', 'Video title', 'Dominant_Topic_Keywords']].head())

    # Identify trending videos
    #enhanced_performance_metrics = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\enhanced_performance_metrics_with_topics.csv', encoding='ISO-8859-1')
    popular_videos = identify_popular_videos(performance_metrics)
    popular_videos.to_csv('popular_videos.csv', index=False)
    print("Printing popular videos")
    print(popular_videos.head())
    print("===============================")

if __name__ == '__main__':
    main()
