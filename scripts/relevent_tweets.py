#usage: python relevent_topics.py date 'key, word, list' topic_name
from gensim.models import HdpModel

import pandas as pd
import re

PROCESSED_PATH = ''
DATA_SUFFIX = '.tsv'
SEP = '\t'

MODEL_PATH = ''
ASSIGNED_PATH = ''
TOPN= 30 # number of words to look through when deciding if a topic is relevent

PROB_THRESHOLD = 0.25 # lower bound on percent probability of being assigned to a given topic

SAVE_PATH ='' #path for saving relevent tweets

def get_relevent_topics(model, keywords, topn=None, weight_threshold=None):
    """
    Takes HDP model and keywords along with one of either topn or weight_threshold to determine relevence

    arguments:
        model - gensim HdpModel
        keywords - list of strings of words we want to look for, should be stemmed
        topn - int, number of words allowed to look through for topn selection
        weight_threshold - float, threshold to cut off for relevence, word weights must be strictly greater than

    returns:
        relevent_topics - list of topic indices relevent to given metric
    """
    if topn is None and weight_threshold is None:
        raise ValueError('One of topn or weight_threshold required')

    topic_term = model.get_topics() #topic term matrix of weights num_topics x num terms
    keywords = np.array(hdp.id2word.doc2idx(keywords)) #makes keywords into id format
    relevent_topics = []
    i= 0
    for topic in topic_term:
        if topn is not None:
            top = np.argsort(topic)[-topn:]
            if pd.Series(keywords).isin(top).any():
                relevent_topics.append(i)
        else:
            eligible = np.argwhere(topic > weight)
            if pd.Series(keywords).isin(eligible).any():
                relevent_topics.append(i)

        i+=1

    return relevent_topics

def is_relevent(topic_dist, relevent_topics):

    topic_dist = np.array(topic_dist)

    topics = topic_dist[:, 0]
    probs = topic_dist[:, 1]

    return (probs[relevent_topics.isin(topics)] >= PROB_THRESHOLD).any()

date = sys.argv[1]
keywords = sys.argv[2]
topic_name = sys.argv[3]

keywords = re.split('\W+', keywords)
print('loading tweets')
processed_tweets = pd.read_csv(PROCESSED_PATH+date+DATA_SUFFIX, sep=SEP, lineterminator='\n', index_col='tweet_id')
hdp = HdpModel.load(MODEL_PATH+date+'.model')
topic_dists = pd.read_csv(ASSIGNED_PATH+date+DATA_SUFFIX, sep=SEP, lineterminator='\n')
print('done')

relevent_topics=pd.Series(get_relevent_topics(hdp, keywords, topn=TOPN))

print('Testing relevence')
is_relevent = topic_dists.topic_distribution.apply(is_relevent, args=(relevent_topics,))
print('Done')
print('Saving')
relevent_ids = topic_dists.tweet_id[is_relevent]
processed_tweets = processed_tweets.loc[relevent_ids]
processed_tweets.to_csv(SAVE_PATH+date+'-'topic_name+DATA_SUFFIX, sep=SEP, lineterminator='\n')
print('Done')
