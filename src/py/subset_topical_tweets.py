#usage: python3 subset_topical_tweets.py date [key,word,list,two-words] topic_name
from gensim.models import HdpModel
import pandas as pd
import re
from pathlib import Path
import sys
import numpy as np

DATA_PATH = '/home/sentiment/data_lake/twitter/nlp_project_samples/'
MODEL_PATH = 'models/daily_topics/'
ASSIGNED_PATH ='data/daily_topic_distributions/'
SAVE_PATH ='data/topical_tweets/' #path for saving relevant tweets

TOPN= 30 # number of words to look through when deciding if a topic is relevant
PROB_THRESHOLD = 0.25 # lower bound on percent probability of being assigned to a given topic

def get_relevant_topics(model, keywords, topn=None, weight_threshold=None):
    """
    Takes HDP model and keywords along with one of either topn or weight_threshold to determine relevence
    arguments:
        model - gensim HdpModel
        keywords - list of strings of words we want to look for, should be stemmed
        topn - int, number of words allowed to look through for topn selection
        weight_threshold - float, threshold to cut off for relevence, word weights must be strictly greater than
    returns:
        relevant_topics - list of topic indices relevant to given metric
    """
    if topn is None and weight_threshold is None:
        raise ValueError('One of topn or weight_threshold required')
    topic_term = model.get_topics() #topic term matrix of weights num_topics x num terms
    keywords = np.array(hdp.id2word.doc2idx(keywords)) #makes keywords into id format
    relevant_topics = []
    i= 0
    for topic in topic_term:
        if topn is not None:
            top = np.argsort(topic)[-topn:]
            if pd.Series(keywords).isin(top).any():
                relevant_topics.append(i)
        else:
            eligible = np.argwhere(topic > weight_threshold)
            if pd.Series(keywords).isin(eligible).any():
                relevant_topics.append(i)

        i+=1
    return relevant_topics

def is_relevant(topic_dist, relevant_topics):
    topic_dist = np.array(topic_dist)
    topics = []
    for i in range(topic_dist.shape[0]):
        if topic_dist[:,1][i] >= PROB_THRESHOLD:
            topics.append(topic_dist[:,0][i])
    return bool(set(topics) & set(relevant_topics))

if __name__ == '__main__':

    date = sys.argv[1]
    keywords = sys.argv[2][1:-1].replace('-', ' ').split(',')
    topic_name = sys.argv[3]

    Path(SAVE_PATH+topic_name).mkdir(exist_ok=True)
    with open(SAVE_PATH+topic_name+'/'+'README.md', 'w') as text_file:
        print('Subsets created using keywords %s' % sys.argv[2], file=text_file)

    print('\nSubsetting data for %s by Topic Modeling' % date)
    tweets = pd.read_csv(DATA_PATH+date+'.tsv', sep='\t', lineterminator='\n')
    tweets = tweets[tweets['tweet_text_stemmed'].notnull()]
    tweets.drop_duplicates('tweet_id', keep=False, inplace=True)

    hdp = HdpModel.load(MODEL_PATH+date+'_topics.model')
    topic_dists = pd.read_csv(ASSIGNED_PATH+date+'.tsv', sep='\t', lineterminator='\n')
    topic_dists = topic_dists[topic_dists['probability']>=PROB_THRESHOLD].reset_index(drop=True)

    relevant_topics = get_relevant_topics(hdp, keywords, topn=TOPN)
    relevant_ids = list(set(topic_dists[topic_dists['topic'].isin(relevant_topics)]['tweet_id'].values))
    subset = tweets[tweets['tweet_id'].isin(relevant_ids)]

    subset.to_csv(SAVE_PATH+topic_name+'/'+date+'_from_topic_model.tsv', sep='\t', index=False)
    print('Done: %s topical tweets' % subset.shape[0])
    del tweets, subset, topic_dists, hdp


    print('\nSubsetting data for %s by String Matching' % date)
    tweets = pd.read_csv(DATA_PATH+date+'.tsv', sep='\t', lineterminator='\n')
    tweets = tweets[tweets['tweet_text_stemmed'].notnull()]
    tweets.drop_duplicates('tweet_id', keep=False, inplace=True)

    subset = pd.DataFrame()
    for keyword in keywords:
        if keyword[0]=='#' or keyword[0]=='@':
            tweets['flag'] = [(re.search('%s\b' % keyword, elem) is not None) for elem in tweets['tweet_text_clean'].values]
        else:
            tweets['flag'] = [(re.search('\b%s\b' % keyword, elem) is not None) for elem in tweets['tweet_text_clean'].values]
        subset = pd.concat([subset, tweets[tweets['flag']==True]].drop('flag', 1))
        tweets = tweets[tweets['flag']==False].reset_index(drop=True).drop('flag', 1)

    subset.to_csv(SAVE_PATH+topic_name+'/'+date+'_from_string_match.tsv', sep='\t', index=False)
    print('Done: %s topical tweets' % subset.shape[0])
    del tweets, subset
