import os
import argparse

from kmeans_utils import *


# usage: python kmeans_for_server.py 5,8,10,15 \
# '/home/sentiment/MIT-NLP-project/data/topical_tweets/chinese_virus/' \
# '/home/sentiment/MIT-NLP-project/figures/kmeans/chinese_virus/' -- date_range 20200101-20200301 --topic chinese_virus

def find_files_in_date_range(date_range, datapath):
    assert date_range.count('-')==1, 'date_range must be separated by one and only one dash'
    start, end = date_range.split('-')
    for f in os.listdir(datapath):
        date = f.split('_')[0]
        if date < start or date > end:
            continue
        
        df = pd.read_csv(os.path.join(datapath, date+'_from_string_match.tsv'), lineterminator='\n', sep='\t')
        np_corpus_embeddings = np.load(os.path.join(datapath, date+'_from_string_match_keyword_normed_embeddings.npy'))
        yield date, df, np_corpus_embeddings

def find_files_in_date_list(date_list, datapath):
    for date in date_list:
        df = pd.read_csv(os.path.join(datapath, date+'_from_string_match.tsv'), lineterminator='\n', sep='\t')
        np_corpus_embeddings = np.load(os.path.join(datapath, date+'_from_string_match_keyword_normed_embeddings.npy'))
        yield date, df, np_corpus_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('list_of_k', 
                        help='what values of k to try (int separated by comma, no space in-between')
    parser.add_argument('datapath',
                        help='path to folder with topic_embeddings')
    parser.add_argument('figure_folder', 
                        help='path to store figures')
    parser.add_argument('--date_range',  # format '20200209-20200210', inclusive on both sides
                        help='will process these dates')
    parser.add_argument('--date_list',  # format '20200209,20200210,20200211', inclusive on both sides
                        help='will process these dates')
    parser.add_argument('--method', default='pca', type=str,
                        help='data transformation before kmeans')
    parser.add_argument('--topic', default='default', type=str,
                        help='topic to decide figure storage folder')
    parser.add_argument('--dims', default=30, type=int,
                        help='number of dimensions for PCA preprocessing before kmeans')

    
    args = parser.parse_args()    
    range_n_clusters = _check_int(args.list_of_k.split(','))

    if args.date_list is None:
        for date, df, np_corpus_embeddings in find_files_in_date_range(args.date_range, args.datapath):
       
            embeddings = transform_embedding(np_corpus_embeddings, args.method, n_components=args.dims)
            if args.method == 'pca':
                plot_dir = os.path.join(args.figure_folder, args.topic, args.method, str(args.dims), date)
            else:
                plot_dir = os.path.join(args.figure_folder, args.topic, args.method, date)

            inertia_dict = plot_silhouette(range_n_clusters, embeddings, plot_dir=plot_dir)
            plot_kmeans_inertia(inertia_dict, plot_dir=plot_dir)
            save_inertia(inertia_dict, plot_dir=plot_dir)
    else:
        date_list = args.date_list.split(',')
        for date, df, np_corpus_embeddings in find_files_in_date_list(date_list, args.datapath):
       
            embeddings = transform_embedding(np_corpus_embeddings, args.method, n_components=args.dims)
            if args.method == 'pca':
                plot_dir = os.path.join(args.figure_folder, args.topic, args.method, str(args.dims), date)
            else:
                plot_dir = os.path.join(args.figure_folder, args.topic, args.method, date)

            inertia_dict = plot_silhouette(range_n_clusters, embeddings, plot_dir=plot_dir)
            plot_kmeans_inertia(inertia_dict, plot_dir=plot_dir)
            save_inertia(inertia_dict, plot_dir=plot_dir)
