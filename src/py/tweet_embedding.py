#Usage: python3 src/py/tweet_embedding.py 20200101 topic_name
import sys
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import argparse

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
TOPICAL_PATH = 'data/topical_tweets/' #where do the relevent tweets live?

def save_embedding(corpus, subset_name, args):
    embeddings = model.encode(corpus, show_progress_bar=True, batch_size=args.batch_size)
    np.save(TOPICAL_PATH+args.topic_name+'/'+args.date+subset_name+'_embeddings.npy', np.array(embeddings))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('date', help='what date?')
    parser.add_argument('topic_name', help='what is the topic name?')
    parser.add_argument('--batch_size', default = 100, type = int, help='batch size')
    parser.add_argument('--topic_model', default = False, type = bool, help='do we also run the version from topic modeling?')
    parser.add_argument('--remove_keyword', default = False, type = bool, help='do we also run the version without the keyword?')
    args = parser.parse_args()

    with open(TOPICAL_PATH+args.topic_name+'/README.md', 'r') as file:
        keywords = file.read().strip()
    keywords = keywords[keywords.find('['):][1:-1].replace('-', ' ').split(',')
    keywords.sort(key=len, reverse=True)
    keyword_regex = r'\b'+r'\b|\b'.join(keywords)+r'\b'
    keyword_regex = keyword_regex.replace(r'\b@', '@').replace(r'\b#', '#')

    print("\nEmbedding data for {}".format(args.date))

    print("From string match")
    corpus = pd.read_csv(TOPICAL_PATH+args.topic_name+'/'+args.date+'_from_string_match.tsv', sep='\t', lineterminator='\n')['tweet_text_clean'].fillna('').values
    save_embedding(corpus, subset_name='_from_string_match', args=args)
    if args.remove_keyword:
        print("From string match removing keywords")
        corpus = [re.sub(keyword_regex, '', elem).strip() for elem in corpus]
        save_embedding(corpus, subset_name='_from_string_match_keyword_normed', args=args)
    del corpus

    if args.topic_model:
        print("From topic model")
        corpus = pd.read_csv(TOPICAL_PATH+args.topic_name+'/'+args.date+'_from_topic_model.tsv', sep='\t', lineterminator='\n')['tweet_text_clean'].fillna('').values
        save_embedding(corpus, subset_name='_from_topic_model', args=args)
        if args.remove_keyword:
            print("From topic model removing keywords")
            corpus = [re.sub(keyword_regex, '', elem).strip() for elem in corpus]
            save_embedding(corpus, subset_name='_from_topic_model_keyword_normed', args=args)
        del corpus

    print("Done with day {} !".format(args.date))
