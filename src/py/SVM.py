import os
import argparse
import pickle
import logging

import pandas as pd
import numpy as np

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.externals import joblib

LR_COLS = ['tweet_id', 'label', 'tfidf']

key_word_dict = {
    'donaldtrump_lax': ['trump'],
    'donaldtrump_strict': ['donaldtrump', 'donald_trump'],
    'NBA': ['nba'],
    'china': ['chinese', 'china'],
    'election': ['democratic primary', 'democratic candidate', ' election', ' elect']
}

def read_data(data_path):
    # data_path = '../../data/supervised_training_set/donaldtrump_lax'
    topic = data_path.split('/')[-1]
    if topic in key_word_dict:
        keywords = key_word_dict[topic]
    else:
        logging.warning('topic: %s not recognized' % topic)
        keywords = []

    train = pd.read_csv(data_path + '/train.csv')
    train = train[pd.notnull(train.tweet_text_stemmed)]
    print('train data shape', train.shape)
    train = remove_topic_words(train, keywords)

    test = pd.read_csv(data_path + '/test.csv')
    test = test[pd.notnull(test.tweet_text_stemmed)]
    print('test data shape', test.shape)
    test = remove_topic_words(test, keywords)
    
    return train, test

def remove_topic_words(df, keywords):
    df['bow'] = df['tweet_text_stemmed']

    if len(keywords)==0:
        return df

    for key in keywords:
        df['bow'] = df['bow'].apply(lambda x: x.replace(key, ''))

    df['bow'] = df['bow'].apply(lambda x: x.replace('  ', ' '))
    return df

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
    df['tfidf'] = df.bow.apply(bow_to_tfidf)
    return df

def train_model_LR(train_df, max_iter=200, C=1.):
    print(C)
    X = train_df.tfidf.tolist()
    y = train_df.label

    clf = LogisticRegression(solver='lbfgs', max_iter=max_iter, C=C).fit(X, y)
    print('training set accuracy:' ,clf.score(X, y))
    return clf

def test_model_LR(test_df, clf, data_folder, save=False):
    test_pred = clf.predict(test_df.tfidf.tolist())
    correct = sum(test_pred==test_df.label)
    print('got %s out of %s correct, accuracy rate is %s' % 
                (correct, 
                test_df.shape[0], 
                correct/test_df.shape[0]))

    # save model and prediction
    if save:
        joblib.dump(clf, data_folder + '/LR.pkl') 

        test_df['LR_label'] = test_pred
        try:
            test_result = test_df.drop('tfidf', axis=1)
        except:
            test_result = test_df
        test_result.to_csv(data_folder + '/test_result.csv', index=False)    

def train_model_SVM(train_df, C=1., kernel='rbf'):
    print(C)
    X = train_df.tfidf.tolist()
    y = train_df.label

    clf = SVC(gamma='auto', C=C, kernel=kernel).fit(X, y)
    print('training set accuracy:' ,clf.score(X, y))
    return clf

def test_model_SVM(test_df, clf, data_folder, save=False):
    test_pred = clf.predict(test_df.tfidf.tolist())
    correct = sum(test_pred==test_df.label)
    print('test set got %s out of %s correct, accuracy rate is %s' % 
                (correct, 
                test_df.shape[0], 
                correct/test_df.shape[0]))
    # save model and prediction
    if save:
        joblib.dump(clf, data_folder + '/SVC.pkl') 
        test_df['SVC_label'] = test_pred
        try:
            test_result = test_df.drop('tfidf', axis=1)
        except:
            test_result = test_df
        test_result.to_csv(data_folder + '/SVC_test_result.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder',
                        help='path to folder with train and test datasets')
    parser.add_argument('tfidf_folder',
                        help='path to folder with tfidf matrix and word dict')
    parser.add_argument('model',
                        help='which type of model to train: SVM, LR')

    parser.add_argument('--skip', action='store_true',
                        help='path to folder with tfidf matrix and word dict')
    parser.add_argument('--save_data', action='store_true',
                        help='save bow tfidf created')
    parser.add_argument('--max_iter', type=int, default=200,
                        help='number of max iterations for model fitting')
    parser.add_argument('--reg', type=float, default=1.,
                        help='inverse regularization stregnth, smaller values specify stronger regularization.')
    parser.add_argument('--kernel', type=str, default='rbf',
                        help='SVM kernel')


    
    args = parser.parse_args()
    # print(os.getcwd())

    if args.skip:
        train = pd.read_csv(args.data_folder + '/train_bow.csv')
        test = pd.read_csv(args.data_folder + '/test_bow.csv')

    else:
        tfidf_matrix, data_dict = read_tfidf(args.tfidf_folder)
        train, test = read_data(args.data_folder)

        test = create_tfidf(test)
        print('finished processing test set')

        train = create_tfidf(train)
        print('finished processing train set')

        if args.save_data:
        	train[LR_COLS].to_csv(args.data_folder + '/train_bow.csv', index=False)
        	test[LR_COLS].to_csv(args.data_folder + '/test_bow.csv', index=False)

    if args.model == 'SVM':
	    clf = train_model_SVM(train, C=args.reg, kernel=args.kernel)
	    test_model_SVM(test, clf, args.data_folder)
    elif args.model == 'LR':
    	clf = train_model_LR(train, max_iter=args.max_iter, C=args.reg)
    	test_model_LR(test, clf, args.data_folder)
    else:
    	raise NotImplementedError('model not implemented:', args.model)
