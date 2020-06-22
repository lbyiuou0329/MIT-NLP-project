#usage: python3 subset_topical_tweets.py date [key,word,list,two-words] topic_name
from gensim.models import HdpModel
import pandas as pd
import re
from pathlib import Path
import numpy as np
import argparse

DATA_PATH = '/home/sentiment/data_lake/twitter/processed/'
MODEL_PATH = 'models/daily_topics/'
ASSIGNED_PATH ='data/daily_topic_distributions/'
SAVE_PATH ='data/topical_tweets/' #path for saving relevant tweets

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
    keywords = np.array(model.id2word.doc2idx(keywords)) #makes keywords into id format
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

def get_topic_model_subset(tweets, args):
    hdp = HdpModel.load(MODEL_PATH+args.date+'_topics.model')
    topic_dists = pd.read_csv(ASSIGNED_PATH+args.date+'.tsv', sep='\t', lineterminator='\n')
    topic_dists = topic_dists[topic_dists['probability']>=args.prob_threshold].reset_index(drop=True)

    relevant_topics = get_relevant_topics(hdp, args.keywords, topn=args.topn)
    relevant_ids = list(set(topic_dists[topic_dists['topic'].isin(relevant_topics)]['tweet_id'].values))
    subset = tweets[tweets['tweet_id'].isin(relevant_ids)]

    return subset

def get_string_match_subset(tweets, args):
    subset = pd.DataFrame()

    for keyword in args.keywords:
        if keyword[0]=='#' or keyword[0]=='@':
            tweets['flag'] = [(re.search(r'%s\b' % keyword, elem) is not None) for elem in tweets['tweet_text_clean'].values]
        else:
            tweets['flag'] = [(re.search(r'\b%s\b' % keyword, elem) is not None) for elem in tweets['tweet_text_clean'].values]
        subset = pd.concat([subset, tweets[tweets['flag']==True].drop('flag', 1)], axis=0)
        tweets = tweets[tweets['flag']==False].reset_index(drop=True).drop('flag', 1)

    return subset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('date', help='what date?')
    parser.add_argument('keywords', help='on which keywords?')
    parser.add_argument('topic_name', help='what is the topic name?')
    parser.add_argument('--country_subset', default = True, type = bool, help='subset to the US?')
    parser.add_argument('--topn', default = 30, type = int, help='number of words to look through when deciding if a topic is relevant')
    parser.add_argument('--prob_threshold', default = 0.25, type = float, help='lower bound on percent probability of being assigned to a given topic')
    args = parser.parse_args()

    args.keywords = args.keywords[1:-1].replace('-', ' ').split(',')

    Path(SAVE_PATH+args.topic_name).mkdir(exist_ok=True)
    with open(SAVE_PATH+args.topic_name+'/'+'README.md', 'w') as text_file:
        print('Subsets created using keywords %s' % args.keywords, file=text_file)

    tweets = pd.read_csv(DATA_PATH+args.date+'.tsv', sep='\t', lineterminator='\n')
    tweets = tweets[tweets['lang']=='en']
    if args.country_subset:
        tweets = tweets[tweets['country']=="US"]
    tweets = tweets[tweets['tweet_text_stemmed'].notnull()]

    print('\nSubsetting data for %s by Topic Modeling' % args.date)
    subset = get_topic_model_subset(tweets, args)
    subset.to_csv(SAVE_PATH+args.topic_name+'/'+args.date+'_from_topic_model.tsv', sep='\t', index=False)
    print('Done: %s topical tweets' % subset.shape[0])

    print('Subsetting data for %s by String Matching' % args.date)
    subset = get_string_match_subset(tweets, args)
    subset.to_csv(SAVE_PATH+args.topic_name+'/'+args.date+'_from_string_match.tsv', sep='\t', index=False)
    print('Done: %s topical tweets' % subset.shape[0])
