import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from gensim import corpora, models
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt
import numpy as np

"""
Author: Pranali Divekar

Description:
The code processes the input files to extract dominant keywords for each video title using Latent Dirichlet Allocation (LDA). 
It then assigns these keywords to users based on their comment with the videos, creating a user profile that reflects their dominant topic preferences. 
The final user profiles are saved for recommendation generations.

Inputs:
- Comments data (CSV obtained from public YouTube API)
- Aggregated video metrics (CSV obtained from public YouTube API)

Outputs:
- A CSV file containing user profiles with their dominant topic keywords.
"""

# Function to preprocess text data
def preprocess_text(text):
    """
    Preprocesses the input text by converting to lowercase, removing punctuation and numbers, 
    tokenizing, removing stopwords, and lemmatizing the tokens.
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


def main():
    comments = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\Datasets\\Youtube_Video_Dataset\\All_Comments_Final.csv', encoding='ISO-8859-1') 
    metrics_by_video = pd.read_csv('C:\\Users\\prana\\Desktop\\Capstone\\Datasets\\Youtube_Video_Dataset\\Aggregated_Metrics_By_Video.csv', encoding='ISO-8859-1')

    # Merge comments with video titles based on video ID
    comments_with_titles = pd.merge(comments, metrics_by_video[['Video', 'Video title']], left_on='VidId', right_on='Video')
    # Preprocess video titles
    comments_with_titles['processed_titles'] = comments_with_titles['Video title'].apply(preprocess_text)

    texts = [simple_preprocess(title) for title in comments_with_titles['processed_titles']]
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Apply LDA
    lda_model = models.LdaMulticore(corpus=corpus, num_topics=10, id2word=dictionary, passes=10, workers=2)
    
    # Extract topic keywords from the LDA model
    topic_keywords = {i: [word for word, prob in lda_model.show_topic(i)] for i in range(lda_model.num_topics)}
    
    # Assign topic keywords to each comment
    comments_with_titles['Dominant_Topic_Keywords'] = [', '.join(topic_keywords[sorted(lda_model[corpus[i]], key=lambda x: (x[1]), reverse=True)[0][0]]) for i in range(len(comments_with_titles))]
    
    # Create user profiles
    user_profiles = comments_with_titles.groupby('user_ID')['Dominant_Topic_Keywords'].apply(lambda x: x.mode()[0]).reset_index()
    
    # Print or save the user profiles
    print(user_profiles.head())
    user_profiles.to_csv('C:\\Users\\prana\\Desktop\\Capstone\\user_profiles.csv', index=False) 



if __name__ == '__main__':
    main()