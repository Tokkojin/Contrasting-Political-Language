from pprint import pprint
import newspaper
from newspaper import Article, news_pool
import collections
import logging
import gensim
import codecs
import os
import re
import sys
import glob

from pathlib import Path

from itertools import count

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import numpy as np

def word_tokenizer(text):
        #tokenizes and stems the text
        tokens = word_tokenize(text)
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
        return tokens


def cluster_texts(texts, nb_of_clusters=5):
        tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
                                        stop_words=stopwords.words('english'),
                                        max_df=0.9,
                                        min_df=0.1,
                                        lowercase=True)
        #builds a tf-idf matrix for the sentences
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        kmeans = KMeans(n_clusters=nb_of_clusters)
        kmeans.fit(tfidf_matrix)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
                clusters[label].append(i)
        return dict(clusters)

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

def numbers( path ):
    for filename in os.listdir(path):
        name, _ = os.path.splittext()
        yield int(name[4:])

if __name__ == '__main__':
    # load nltk's English stopwords as variable called 'stopwords'
    # stopwords = nltk.corpus.stopwords.words('english')

    # # load nltk's SnowballStemmer as variabled 'stemmer'
    # stemmer = SnowballStemmer("english")

    # #define vectorizer parameters
    # tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
    #                              min_df=0.2, stop_words='english',
    #                              use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

    # tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses

    # print(tfidf_matrix.shape)

    # cnn_article = Article(url="http://www.cnn.com/2017/07/30/politics/pence-russia-north-korea-trump/index.html", language='en')
    # nyt_article = Article(url="https://www.nytimes.com/2017/04/17/world/asia/trump-north-korea-nuclear-us-talks.html", language='en')

    # stops = set(stopwords.words("english"))

    # cnn_clean = import_and_clean(cnn_article, stops)
    # nyt_clean = import_and_clean(nyt_article, stops)

    dir = os.path.dirname(os.getcwd())
    print(dir)
    # gnews_vector_path = os.path.join(dir, 'pol_lang/GoogleNews-vectors-negative300.bin')
    # model = gensim.models.KeyedVectors.load_word2vec_format(gnews_vector_path, binary=True)
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # print("sim(cnn_clean, nyt_clean) = ", w2v(cnn_clean, nyt_clean, model),"/1.")

    # wash_post_paper = newspaper.build('https://www.washingtonpost.com/politics/?nid=top_nav_politics')

    # cnn_paper       = newspaper.build('http://www.cnn.com/politics', memoize_articles=False)
    # cnn_paper.download()
    # cnn_paper.parse()
    # fox_paper       = newspaper.build('http://www.foxnews.com/politics.html', memoize_articles=False)
    # fox_paper.download()
    # fox_paper.parse()   

    # # papers = [cnn_paper, fox_paper]
    # # news_pool.set(papers, threads_per_source=2)
    # # news_pool.join()

    # # print(cnn_paper.size())
    # # print(fox_paper.size())

    path_to_articles = os.path.join(dir, 'pol_lang/articles')
    print(path_to_articles)
    # if not os.path.exists(path_to_articles): os.makedirs(path_to_articles)

    # # determine what file name count should be 
    # if not numbers(path_to_articles):
    #     count = max(numbers(path_to_articles))
    #     count += 1
    # else:
    #     count = 1

    # for article in fox_paper.articles:
    #     article.download()
    #     article.parse()

    #     filename = 'text' + str(count) + '.txt'
    #     complete_filename = os.path.join(path_to_articles, filename)
    #     print(complete_filename)
        
    #     file = open(complete_filename, 'w')
    #     file.write(article.text)

    #     file.close()

    #     count += 1
    
    texts = []
    files = [file for file in glob.glob(path_to_articles + '/**/*.txt', recursive=True)]
    for file in files:
        texts.append(Path(file).read_text())
    nclusters= 3
    clusters = cluster_texts(texts, nclusters)
    for cluster in range(nclusters):
            print ("cluster ",cluster,":")
            for i,sentence in enumerate(clusters[cluster]):
                    print ("\tsentence ",i,": ",texts[sentence][:50])