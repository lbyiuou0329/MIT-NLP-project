# usage: python3 src/twitter_data_extraction.py test --incl_keywords trump --country PRT --lang en --start_date 2019-01-01 --end_date 2019-01-02 --text_path /home/sentiment/data-lake/twitter/processed/ --geo_path /home/sentiment/data-lake/twitter/geoinfo/

import pandas as pd
import argparse
from tqdm.auto import tqdm
from utils.extraction_utils import get_dates, get_data, get_embeddings

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

    embeddings = get_embeddings(df['tweet_text_clean'].fillna('').values, args=args)
