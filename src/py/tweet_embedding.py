#Usage: python tweet_embedding.py date topic_name
import sys
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

TOPICAL_PATH = 'data/topical_tweets/' #where do the relevent tweets live?
BATCH_SIZE = 100

def save_embedding(corpus, subset_name):
    embeddings = model.encode(corpus, show_progress_bar=True, batch_size=BATCH_SIZE)
    np.save(TOPICAL_PATH+topic_name+'/'+date+subset_name+'_embeddings.npy', np.array(embeddings))

if __name__ == '__main__':

    date = sys.argv[1]
    topic_name = sys.argv[2]

    with open(TOPICAL_PATH+topic_name+'/README.md', 'r') as file:
        keywords = file.read().strip()
    keywords = keywords[keywords.find('['):][1:-1].replace('-', ' ').split(',')
    keywords.sort(key=len, reverse=True)
    keyword_regex = r'\b'+r'\b|\b'.join(keywords)+r'\b'
    keyword_regex = keyword_regex.replace(r'\b@', '@').replace(r'\b#', '#')


    print("\nEmbedding data for %s" % date)

    print("From topic model")
    corpus = pd.read_csv(TOPICAL_PATH+topic_name+'/'+date+'_from_topic_model.tsv', sep='\t', lineterminator='\n')['tweet_text_clean'].fillna('').values
    save_embedding(corpus, subset_name='_from_topic_model')
    print("From topic model removing keywords")
    corpus = [re.sub(keyword_regex, '', elem).strip() for elem in corpus]
    save_embedding(corpus, subset_name='_from_topic_model_keyword_normed')
    del corpus

    print("From string match")
    corpus = pd.read_csv(TOPICAL_PATH+topic_name+'/'+date+'_from_string_match.tsv', sep='\t', lineterminator='\n')['tweet_text_clean'].fillna('').values
    save_embedding(corpus, subset_name='_from_string_match')
    print("From string match removing keywords")
    corpus = [re.sub(keyword_regex, '', elem).strip() for elem in corpus]
    save_embedding(corpus, subset_name='_from_string_match_keyword_normed')
    del corpus
