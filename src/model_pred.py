import os
import argparse
import logging

import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from scipy.sparse import load_npz
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
from sklearn.externals import joblib


# topic_dict = {
#   'trump': 'donaldtrump_strict',
#   'nba': 'NBA',
#   'china': 'china',
#   'election': 'election'
# }

key_word_dict = {
    'donaldtrump_lax': ['@realdonaldtrump', 'donaldtrump', 'donald_trump', 'trump'],
    'donaldtrump_strict': ['@realdonaldtrump', 'donaldtrump', 'donald_trump'],
    'NBA': ['nba'],
    'china': ['chinese', 'china'],
    'election': ['democratic primary', 'democratic candidate', ' election', ' elect']
}

def read_data(data_path):
    # data_path = '../../data/supervised_training_set/donaldtrump_lax'
    data_path = data_path.replace('supervised_training_set/', '') + 'processed/'
    # import pdb; pdb.set_trace()
    df_file = data_path+'2020-01-01-en-US_processed.tsv'
    df = pd.read_csv(df_file, sep='\t', encoding='utf8', engine='c', lineterminator='\n')
    df = df[pd.notnull(df.tweet_text_stemmed)]
    print('raed in data of shape', df.shape)   
    return df

def read_data_generator(data_path, chunksize=10000):
    # data_path = '../../data/supervised_training_set/donaldtrump_lax'
    data_path = data_path.replace('supervised_training_set/', '') + 'processed/'
    
    df_file = data_path+'2020-01-01-en-US_processed.tsv'
    for data in pd.read_csv(df_file, sep='\t', encoding='utf8', engine='c', lineterminator='\n', chunksize=chunksize):
        data = data[pd.notnull(data.tweet_text_stemmed)]
        yield data    

def read_tfidf(tfidf_folder):
    tfidf_matrix = load_npz(tfidf_folder + '/tfidf_total.npz')
    # tfidf_matrix = tfidf_matrix.T
    print('matrix shape:', tfidf_matrix.shape)

    with open(tfidf_folder+'/tfidf_words_total.txt', 'r') as txt:
        data = txt.read().split('\n')
    print('number of words:', len(data))
    data_dict = {word:i for i, word in enumerate(data[:-1])}    
    return tfidf_matrix, data_dict

def bow_to_sparse(bow):
    elements = bow.split(' ')
    locs = []
    for ele in elements:
        try:
            loc = data_dict[ele]
            locs.append(loc)
        except KeyError:
            continue
    return locs

def bow_to_tfidf(bow):
    locs = bow_to_sparse(bow)
    bow_vector = np.asarray(np.sum(tfidf_matrix[locs], axis=0))
    return bow_vector.squeeze()

def create_tfidf(df):
    df['tfidf'] = df.tweet_text_stemmed.apply(bow_to_tfidf)
    return df

def _pred_str_match(topic, df):
    key_words = key_word_dict[topic]
    y_pred = pd.Series([False]*df.shape[0])
    for key in key_words:
        y_pred = y_pred | df['tweet_text_stemmed'].apply(
            lambda x: key in x).reset_index(drop=True)

    return y_pred, pd.Series([0]*df.shape[0])

def _pred_supervised(df, clf):
    df = create_tfidf(df)
    X = df.tfidf.tolist()
    y_pred = clf.predict(X)
    y_log_proba = clf.predict_log_proba(X)[:, 1] # only store log prob of true 
    return y_pred, y_log_proba

def predict(topic, method, df, data_folder=None):
    if method == 'str_match':
        y_pred, y_log_proba = _pred_str_match(topic, df)
    elif method == 'LR':
        assert data_folder is not None
        clf = joblib.load(os.path.join(data_folder, topic, 'LR.pkl'))
        y_pred, y_log_proba = _pred_supervised(df, clf)
    elif method == 'SVC':
        assert data_folder is not None
        clf = joblib.load(os.path.join(data_folder, topic, 'SVC.pkl'))
        y_pred, y_log_proba = _pred_supervised(df, clf)
    else:
        raise ValueError('invalid method: %s' % method)

    return y_pred, y_log_proba

def save_overall_result(data_folder, topic, method, override=False):
    topic_name = topic + '_' + method

    if os.path.isfile(data_folder + '/overall_result.csv'):
        print('reading old overall_result')
        overall_result = pd.read_csv(data_folder + '/overall_result.csv')
        
        if override or (topic_name not in overall_result.columns):
            overall_result[topic_name] = y_pred_store
            overall_result[topic_name+'_log_prob'] = y_log_proba_store
    else:
        print('creating new overall_result')
        df_data = {'tweet_id': tweet_id_store,
                    topic_name: y_pred_store,
                    topic_name+'_log_prob': y_log_proba_store}
        overall_result = pd.DataFrame(data=df_data)

    overall_result.to_csv(data_folder + '/overall_result.csv', index=False)
    return overall_result

def save_predictions(overall_result, topic, data_folder, model):
    topic_name = topic + '_' + model
    topic_df = overall_result[overall_result[topic_name]]
    tweets = read_data(data_folder)
    topic_df = pd.merge(topic_df, tweets, on='tweet_id', how='left')
    topic_df.to_csv(os.path.join(data_folder, topic, '%s_predicted_tweets.csv' % model), index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder',
                        help='path to folder with train and test datasets')
    parser.add_argument('tfidf_folder',
                        help='path to folder with tfidf matrix and word dict')
    parser.add_argument('topic',
                        help='which topic to predict: trump, nba, china, election')
    parser.add_argument('--model', default='LR', type=str,
                        help='which type of model to train: SVM, LR, str_match')
    parser.add_argument('--save_data', action='store_true',
                        help='save tweets predicted')
    parser.add_argument('--override', action='store_true',
                        help='override old predictions')
    
    args = parser.parse_args()
    # print(os.getcwd())

    y_pred_store = []
    y_log_proba_store = []
    tweet_id_store = []

    # import pdb; pdb.set_trace()
    tfidf_matrix, data_dict = read_tfidf(args.tfidf_folder)

    for df_chunk in tqdm(read_data_generator(args.data_folder)):
        print('processing %s rows of data' % df_chunk.shape[0])

        y_pred, y_log_proba = predict(args.topic, args.model, df_chunk, data_folder=args.data_folder)
        
        y_pred_store.extend(y_pred)
        y_log_proba_store.extend(y_log_proba) 
        tweet_id_store.extend(df_chunk.tweet_id)
        
        # break

    overall_result = save_overall_result(args.data_folder, args.topic, args.model, override=args.override)

    if args.save_data:
        save_predictions(overall_result, args.topic, args.data_folder, args.model)
