#usage: python3 subset_topical_tweets.py date [key,word,list,two-words] topic_name
# --text_path '/home/sentiment/data_lake/twitter/processed/'

import pandas as pd
import re
from pathlib import Path
import numpy as np
import argparse
import sys
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from utils.extraction_utils import get_string_match_subset, get_topic_model_subset, get_dates

def save_embedding(corpus, args):
    embeddings = model.encode(corpus, show_progress_bar=True, batch_size=args.batch_size)
    np.save('data/topical_tweets/{}{}_embeddings.npy'.format(args.ext_name, args.suffix), np.array(embeddings))

def get_data(date, args):

    if True:#    try:
        text = pd.read_csv(args.text_path+"{}{}{}.tsv.gz".format(date.year, str(date.month).zfill(2), str(date.day).zfill(2)), sep='\t', usecols=['tweet_id', 'lang', 'tweet_text_clean', 'tweet_text_keywords'])
        if args.lang is not None:
            text = text[text['lang']==args.lang.lower()]
        text = text[text['tweet_text_keywords'].notnull()]

        geo = pd.read_csv(args.geo_path + '{}-{}-{}.tsv.gz'.format(date.year, str(date.month).zfill(2), str(date.day).zfill(2)), sep=',')
        if args.country is not None:
            geo = geo[geo['country']==args.country.upper()]

        tweets = pd.merge(text, geo, how='inner', on='tweet_id')
        del text, geo

    else:#except:
        print("\nNo data for {}.".format(date))
        return pd.DataFrame()

    if args.topic_model:
        subset = get_topic_model_subset(tweets, args)
        print('Done: %s topical tweets' % subset.shape[0])
    else:
        subset = get_string_match_subset(tweets, args)
        print('Done: %s topical tweets' % subset.shape[0])

    return subset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('ext_name', help='what is the extraction name?')
    parser.add_argument('--suffix', default = '', type=str, help='suffix to file names')

    ## Subsetting
    parser.add_argument('--incl_keywords', nargs='*', default='', help='Which keywords do you want to include in the extraction?')
    parser.add_argument('--excl_keywords', nargs='*', default='', help='Which keywords do you want to exclude in the extraction?')
    parser.add_argument('--country', default = None, type = str, help='subset to a specific country?')
    parser.add_argument('--lang', default = None, type = str, help='subset to a specific language?')

    ## Date range
    parser.add_argument('--start_date', default='2019-01-01', help='what start date?')
    parser.add_argument('--end_date', default='2019-12-31', help='what end date?')

    ## Paths
    parser.add_argument('--text_path', default='', help='path to twitter text data')
    parser.add_argument('--geo_path', default='', help='path to twitter geography data')

    ## Topic Model
    parser.add_argument('--topic_model', default = False, type = bool, help='do we also run the version from topic modeling (or just string matching)?')
    parser.add_argument('--topn', default = 30, type = int, help='number of words to look through when deciding if a topic is relevant')
    parser.add_argument('--prob_threshold', default = 0.25, type = float, help='lower bound on percent probability of being assigned to a given topic')

    ## Embeddings
    parser.add_argument('--model', default = 'distilbert-base-nli-stsb-mean-tokens', type = str, help='embedding model')
    parser.add_argument('--batch_size', default = 100, type = int, help='batch size')
    parser.add_argument('--remove_keyword', default = False, type = bool, help='do we also run the version without the keyword?')
    args = parser.parse_args()

    args.incl_keywords = list(args.incl_keywords)
    args.excl_keywords = list(args.excl_keywords)

    if args.topic_model:
        args.suffix += "_from_topic_model"

    with open('data/topical_tweets/{}{}_README.md'.format(args.ext_name, args.suffix), 'w') as text_file:
        print('Subsets created using arguments:\n{}'.format(args), file=text_file)

    dates = get_dates(args)

    df = pd.DataFrame()
    for date in tqdm(dates):
        temp = get_data(date, args)
        df = pd.concat([df, temp], axis=0)

    df.to_csv('data/topical_tweets/{}{}.tsv'.format(args.ext_name, args.suffix), sep='\t', index=False)

    model = SentenceTransformer(args.model)
    save_embedding(df['tweet_text_clean'].fillna('').values, args=args)
