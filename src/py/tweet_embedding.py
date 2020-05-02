#Usage: python tweet_embedding.py date topic_name
import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

TOPICAL_PATH = 'data/topical_tweets/' #where do the relevent tweets live?
BATCH_SIZE = 100

date = sys.argv[1]
topic_name = sys.argv[2]

if __name__ == '__main__':

    print("\nEmbedding data for %s" % date)

    tweets = pd.read_csv(TOPICAL_PATH+topic_name+'/'+date+'.tsv', sep='\t', lineterminator='\n')

    embeddings = model.encode(tweets['tweet_text_clean'].fillna('').values, show_progress_bar=True, batch_size=BATCH_SIZE)

    np.save(TOPICAL_PATH+topic_name+'/'+date+'_embeddings.npy', np.array(embeddings))
