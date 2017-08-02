from pprint import pprint
from newspaper import Article
import logging
import gensim
import os
import sys
import nltk
from nltk.corpus import stopwords
import numpy as np
import re



def w2v(s1,s2,wordmodel):
    if s1==s2:
        return 1.0

    s1words=s1.split()
    s2words=s2.split()
    s1wordsset=set(s1words)
    s2wordsset=set(s2words)
    vocab = wordmodel.vocab #the vocabulary considered in the word embeddings

    if len(s1wordsset & s2wordsset)==0:
            return 0.0
    copy1 = s1wordsset.copy()
    copy2 = s2wordsset.copy()
    for word in copy1: #remove sentence words not found in the vocab
            if (word not in vocab):
                s1wordsset.remove(word)
                print(word + ' has been removed.\n')
    for word in copy2: 
            if (word not in vocab):
                s2wordsset.remove(word)
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

    return(" ".join( article_clean_array ))

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("sim(cnn_clean, nyt_celean) = ", w2v(cnn_clean, nyt_clean, model),"/1.")