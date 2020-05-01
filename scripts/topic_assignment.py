###
#Usage: python topic_assignment.py year-mo-da
# Writes to file tweet_id topic distribution pairs and trained HDP model for date
###
import pandas as pd
import numpy as np
import os
import nltk
import re
import sys
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from collections import Counter, OrderedDict
import tqdm

from gensim.matutils import Sparse2Corpus
from gensim.models import HdpModel
from gensim.corpora import Dictionary
import warnings

#alter these global variables for save paths
DATA_PATH = ''
DATA_SUFFIX='.tsv'
SEP ='\t'
PROCESSED_PATH =''

MODEL_PATH = ''
ASSIGNED_PATH =''


nltk.download("stopwords")
eng_stopwords = stopwords.words('english')

def clean_tweet(string):

    string = string.lower()

    string = re.sub(r'\bhttps?\:\/\/[^\s]+', ' ', string) #remove websites

    # Classic replacements:
    string = re.sub(r'\&', ' and ', string)
    string = re.sub(r'\s\@\s', ' at ', string)
    # replace some user names by real names?

    string = re.sub(r'\s+', ' ', string).strip()

    return string


def process_data(filename):

    print("\nProcessing datafile {}".format(filename))

    df = pd.read_csv(
        filename,
        sep=SEP, lineterminator='\n', low_memory=False,
        usecols=['country', 'lang', 'tweet_id', 'tweet_text', 'reply_to_tweet_id']
    )
    df = df[(df['country']=="US")&(df['lang']=="en")].reset_index(drop=True)

    #restrict to only original posts
    df = df[df['reply_to_tweet_id'].isna()].reset_index(drop=True)

    print("Cleaning...")
    clean_corpus = [clean_tweet(elem) for elem in tqdm(df['tweet_text'].values)]
    print('Done')

    df['clean_tweets'] = clean_corpus

    df.to_csv(PROCESSED_PATH+date+DATA_SUFFIX, sep=SEP, lineterminator='\n')
    tweet_ids = list(temp['tweet_id'].values)

    return tweet_ids, clean_corpus


def create_bag_of_words(corpus, ngram_range=(1,1), stopwords=None, stem=False, min_df=0.05, max_df=0.95, use_idf=False):

    if stem:
        stemmer = nltk.SnowballStemmer("english")
        tokenize = lambda x: [stemmer.stem(i) for i in x.split()]
        stopwords = [stemmer.stem(elem) for elem in stopwords]
    else:
        tokenize = None

    vectorizer = CountVectorizer(
        tokenizer=tokenize,
        ngram_range=ngram_range,
        stop_words = stopwords,
        strip_accents='unicode',
        min_df=min_df,
        max_df=max_df,
        token_pattern='[^\s]+'
    )

    bag_of_words = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names()

    if use_idf:
        transformer = TfidfTransformer(norm = None, smooth_idf = True,sublinear_tf = True)
        bag_of_words = transformer.fit_transform(bag_of_words)

    return bag_of_words, features

def get_topic_distributions(model, texts):
    """
    Returns distributions of topics for every text in texts using hdp model
    Ignores topics below 0.01 probability

    arguments:
        model - gensim HdpModel
        texts - list of lists of (wordid, float) (documents in BoW form)
    returns:
        topic_dist - list of list of (topic_id, probability) pairs for each text
    """

    dists = []

    print('Finding topic dists')
    for t in tqdm(texts):
        topic_dist = model[t]

        dists.append(topic_dist)
    print('Done')
    return dists

date = sys.argv[1]
fn = DATA_PATH+date+DATA_SUFFIX

tweet_ids, clean_corpus = process_data(fn)

print("Creating TFIDF BOWs")
bow_tfidf, features_tfidf = create_bag_of_words(clean_corpus, ngram_range=(1,3), use_idf=True, min_df=0.0001, stem=True, stopwords=eng_stopwords)
print('Done')

corpus = Sparse2Corpus(bow_tfidf, documents_columns=False)
vocab_dict = Dictionary([features_tfidf])

print('Training HDP model')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    hdp = HdpModel(corpus, vocab_dict)

hdp.save(MODEL_PATH+date+'.model')
print('Done')

topic_dist = get_topic_distributions(hdp, corpus)

print('Saving...')
df = pd.DataFrame({'tweet_id': tweet_ids, 'topic_distribution': topic_dist})

df.to_csv(ASSIGNED_PATH+date+DATA_SUFFIX, sep=sep, lineterminator='\n')
print('Done')
