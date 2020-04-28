###
# Usage: python3 topic_assignment.py YYYYMMDD
# Writes to file tweet_id topic distribution pairs and trained HDP model for date
###
import pandas as pd
import numpy as np
import os
import sys
from tqdm.autonotebook import tqdm
from gensim.matutils import Sparse2Corpus
from gensim.models import HdpModel
from gensim.corpora import Dictionary
import warnings

from utils.word_counting import create_bag_of_words

#alter these global variables for save paths
DATA_PATH = '/home/sentiment/data_lake/twitter/nlp_project_samples/'
DATA_SUFFIX='.tsv'
SEP ='\t'
MODEL_PATH = 'models/daily_topics/'
ASSIGNED_PATH ='data/daily_topic_distributions/'
LANGUAGE = 'en'
COUNTRY = 'US'

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

if __name__ == '__main__':

    print("Reading in data for %s" % date)
    df = pd.read_csv(DATA_PATH+date+DATA_SUFFIX, sep=SEP, lineterminator='\n', usecols=['tweet_id', 'country', 'lang', 'tweet_text_stemmed'])
    if COUNTRY != None:
        df = df[df['country']==COUNTRY]
    if LANGUAGE != None:
        df = df[df['lang']!=LANGUAGE]
    df = df[df['tweet_text_stemmed'].notnull()] # Restrict to only
    print("Topic modeling on %s tweets" % df.shape[0])

    print("Creating TFIDF BoW...")
    bow, features = create_bag_of_words(df['tweet_text_stemmed'], ngram_range=(1,3), use_idf=True, min_df=0.0001)
    bow = Sparse2Corpus(bow, documents_columns=False)
    features = Dictionary([features])
    print('Done')

    print('Training HDP model...')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        hdp = HdpModel(bow, features)
    topic_dist = get_topic_distributions(hdp, bow)
    print('Done')

    print('Saving...')
    hdp.save(MODEL_PATH+date+'_topics.model')
    pd.DataFrame({
        'tweet_id': df['tweet_id'],
        'topic_distribution': topic_dist
    }).to_csv(ASSIGNED_PATH+date+DATA_SUFFIX, sep=sep, lineterminator='\n')
    print('Done')

    del hdp, topic_dist, df, bow, features
