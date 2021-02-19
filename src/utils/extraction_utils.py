from gensim.models import HdpModel
import pandas as pd
import re
from pathlib import Path
import numpy as np
from datetime import date, timedelta

def get_dates(args):

    start_date = date(int(args.start_date[:4]), int(args.start_date[5:7]), int(args.start_date[8:]))
    end_date = date(int(args.end_date[:4]), int(args.end_date[5:7]), int(args.end_date[8:]))

    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days +1)]

    return dates

def get_data(date, args):

    try:
        text = pd.read_csv(args.text_path+"{}{}{}.tsv.gz".format(date.year, str(date.month).zfill(2), str(date.day).zfill(2)), sep='\t', usecols=['tweet_id', 'lang', 'tweet_text_clean', 'tweet_text_keywords'])
        if args.lang is not None:
            text = text[text['lang']==args.lang.lower()]
        text = text[text['tweet_text_keywords'].notnull()]

        geo = pd.read_csv(args.geo_path + '{}-{}-{}.tsv.gz'.format(date.year, str(date.month).zfill(2), str(date.day).zfill(2)), sep=',')
        if args.country is not None:
            geo = geo[geo['country']==args.country.upper()]

        tweets = pd.merge(text, geo, how='inner', on='tweet_id')
        del text, geo
    except:
        print("\nNo data for {}.".format(date))
        return pd.DataFrame()

    if args.topic_model:
        subset = get_topic_model_subset(tweets, args)
    else:
        subset = get_string_match_subset(tweets, args)

    subset['date'] = date
    
    return subset

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
    hdp = HdpModel.load('models/daily_topics/'+args.date+'_topics.model')
    topic_dists = pd.read_csv('data/daily_topic_distributions/'+args.date+'.tsv', sep='\t', lineterminator='\n')
    topic_dists = topic_dists[topic_dists['probability']>=args.prob_threshold].reset_index(drop=True)

    relevant_topics = get_relevant_topics(hdp, args.keywords, topn=args.topn)
    relevant_ids = list(set(topic_dists[topic_dists['topic'].isin(relevant_topics)]['tweet_id'].values))
    subset = tweets[tweets['tweet_id'].isin(relevant_ids)]

    return subset

def get_string_match_subset(tweet_text, args):

    if len(args.incl_keywords)>0:
        regex = '|'.join(args.incl_keywords)
        tweet_text['keep'] = [bool(re.search(regex, elem)) for elem in tweet_text['tweet_text_keywords'].values]
        tweet_text = tweet_text[tweet_text['keep']==True].reset_index(drop=True)
        del tweet_text['keep']
    if len(args.excl_keywords)>0:
        regex = '|'.join(args.excl_keywords)
        tweet_text['drop'] = [bool(re.search(regex, elem)) for elem in tweet_text['tweet_text_keywords'].values]
        tweet_text = tweet_text[tweet_text['drop']==False].reset_index(drop=True)
        del tweet_text['drop']

    return tweet_text

def get_embeddings(corpus, args):
    model = SentenceTransformer(args.model)
    embeddings = model.encode(corpus, show_progress_bar=True, batch_size=args.batch_size)
    np.save('data/topical_tweets/{}{}_embeddings.npy'.format(args.ext_name, args.suffix), np.array(embeddings))
    return embeddings
