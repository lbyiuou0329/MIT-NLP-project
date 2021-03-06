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
MIN_DF = 0.0001 # In what share of documents must a word appear?

def get_topic_distributions(model, texts, ids):
    """
    Returns distributions of topics for every text in texts using hdp model
    Ignores topics below 0.01 probability
    arguments:
        model - gensim HdpModel
        texts - list of lists of (wordid, float) (documents in BoW form)
    returns:
        topic_dist - list of list of (topic_id, probability) pairs for each text
    """
    id_list = []
    topic_list = []
    prob_list = []
    print('Finding topic dists...')
    for i in tqdm(range(len(texts))):
        topic_dist = model[texts[i]]
        if len(topic_dist)>0:
            id_list += [ids[i] for elem in topic_dist]
            topic_list += list(np.array(topic_dist)[:,0])
            prob_list += list(np.array(topic_dist)[:,1])
    topic_dists = pd.DataFrame({'tweet_id':id_list, 'topic':topic_list, 'probability':prob_list})
    return topic_dists

date = sys.argv[1]

if __name__ == '__main__':

    print("\nReading in data for %s" % date)
    df = pd.read_csv(DATA_PATH+date+DATA_SUFFIX, sep=SEP, lineterminator='\n', usecols=['tweet_id', 'country', 'lang', 'tweet_text_stemmed'])
    if COUNTRY != None:
        df = df[df['country']==COUNTRY]
    if LANGUAGE != None:
        df = df[df['lang']==LANGUAGE]
    df = df[df['tweet_text_stemmed'].notnull()] # Restrict to only
    print("Topic modeling on %s tweets" % df.shape[0])

    print("Creating TFIDF BoW...")
    bow, features = create_bag_of_words(df['tweet_text_stemmed'], ngram_range=(1,3), use_idf=True, min_df=MIN_DF)
    print('Done: %s features' % len(features))

    print('Training HDP model...')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        hdp = HdpModel(Sparse2Corpus(bow, documents_columns=False), Dictionary([features]))
    topic_dists = get_topic_distributions(hdp, Sparse2Corpus(bow, documents_columns=False), df['tweet_id'].values)
    print('Done')

    print('Saving...')
    hdp.save(MODEL_PATH+date+'_topics.model')
    np.savetxt(MODEL_PATH+date+'_features.txt', features, fmt='%s', delimiter='\n', encoding="utf-8")
    topic_dists.to_csv(ASSIGNED_PATH+date+DATA_SUFFIX, sep=SEP, index=False)
    print('Done')

    del hdp, topic_dist, df, bow, features
