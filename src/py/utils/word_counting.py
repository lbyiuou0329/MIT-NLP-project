import numpy as np
from collections import Counter, OrderedDict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def create_bag_of_words(corpus, ngram_range=(1,1), min_df=0.05, max_df=0.95, use_idf=False):

    vectorizer = CountVectorizer(
        tokenizer = None, # Stemming done in cleaning
        stop_words = None, # Stopwords removed in cleaning
        ngram_range = ngram_range,
        strip_accents = 'unicode',
        min_df = min_df,
        max_df = max_df,
        token_pattern='[^\s]+' # Keep all non-space words
    )

    bag_of_words = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names()

    if use_idf:
        transformer = TfidfTransformer(norm = None, smooth_idf = True,sublinear_tf = True)
        bag_of_words = transformer.fit_transform(bag_of_words)

    return bag_of_words, features

def get_word_counts(bag_of_words, feature_names):

    word_count = np.asarray(np.sum(bag_of_words.toarray(),axis=0)).ravel()
    dict_word_counts = dict(zip(feature_names, word_count))
    orddict_word_counts = OrderedDict(sorted(dict_word_counts.items(), key=lambda x: x[1], reverse=True), )

    return orddict_word_counts
