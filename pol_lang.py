from pprint import pprint
from newspaper import Article
import logging
import gensim
import os
import sys
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

def w2v(s1,s2,wordmodel):
    vocab = wordmodel.vocab

    s1words = s1.copy()
    s2words = s2.copy()

    for word in s1: #remove sentence words not found in the vocab
        if (word not in vocab):
            s1words.remove(word)
            print(word + ' has been removed.\n')
    for word in s2: 
        if (word not in vocab):
            s2words.remove(word)
            print(word + ' has been removed.\n')

    return wordmodel.n_similarity(s1words, s2words)

def import_and_clean(article, stops):
    article.download()
    article.parse()

    # remove non-alphabet characters
    article_clean = re.sub("[^a-zA-Z]", " ", article.text)

    # text into array of words
    article_clean_array = article_clean.lower().split()

    # remove stop words
    article_clean_array = [word for word in article_clean_array if not word in stops] 

    return(article_clean_array)

if __name__ == '__main__':
    cnn_article = Article(url="http://www.cnn.com/2017/07/30/politics/pence-russia-north-korea-trump/index.html", language='en')
    nyt_article = Article(url="https://www.nytimes.com/2017/04/17/world/asia/trump-north-korea-nuclear-us-talks.html", language='en')

    stops = set(stopwords.words("english"))

    cnn_clean = import_and_clean(cnn_article, stops)
    nyt_clean = import_and_clean(nyt_article, stops)

    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform(["I prefer clustering to classification",
    "I prefer visiting my family","Do you prefer clustering to classification too?", 
    "Their family is happy"]) #each sentence can be replaced by a whole document

    dir = os.path.dirname(os.getcwd())
    gnews_vector_path = os.path.join(dir, 'pol_lang/GoogleNews-vectors-negative300.bin')
    model = gensim.models.KeyedVectors.load_word2vec_format(gnews_vector_path, binary=True)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("sim(cnn_clean, nyt_celean) = ", w2v(cnn_clean, nyt_clean, model),"/1.")