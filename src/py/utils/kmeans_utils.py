## usage: python3 src/py/utils/kmeans_utils.py 20200318
## more complicated usage: python3 src/py/utils/kmeans_utils.py 20200318 --subset_mode topic_model --keyword_normed True --list_of_k [3,4,5,7,10] --topic covid --method pca --dims 20

import os
# import sys
# import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

PLOT_DIR = 'figures/kmeans/'
DATA_PATH = 'data/topical_tweets/'

random_state = 123

def _check_int(list_of_k):
    result = [int(k) for k in list_of_k]
    # will throw type error if not int
    return result

<<<<<<< HEAD
def read_in_data(tweet_file, embedding_file):
    df = pd.read_csv(tweet_file, lineterminator='\n', sep='\t')
    embeddings = np.load(embedding_file)
    return df, embeddings

def plot_kmeans_inertia(inertia_dict, plot_dir, filename):
=======
def read_in_data(embedding_file=embedding_file, tweet_file=tweet_file):
    df = pd.read_csv(tweet_file, lineterminator='\n', sep='\t')
    # df['len_text'] = df['tweet_text_clean'].apply(lambda x: len(x.split()))
    np_corpus_embeddings_trump = np.load(embedding_file)
    return df, np_corpus_embeddings_trump

def plot_kmeans_inertia(inertia_dict, plot_dir=PLOT_DIR):
>>>>>>> 7a5aac2dd9d4f9c69e72f8b8068622d6c141fa00
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    keys = sorted(inertia_dict.keys())
    values = [inertia_dict[k] for k in keys]
    plt.plot(keys, values)
<<<<<<< HEAD
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig(os.path.join(plot_dir, filename + '_inertia.jpg'), bbox_inches='tight')
    plt.close()

def save_inertia(inertia_dict, plot_dir, filename):
=======
    plt.savefig(os.path.join(plot_dir, 'inertia_for_%s.jpg' % '|'.join([str(k) for k in keys])))
    plt.close()

def save_inertia(inertia_dict, plot_dir=PLOT_DIR):
>>>>>>> 7a5aac2dd9d4f9c69e72f8b8068622d6c141fa00
    import json
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

<<<<<<< HEAD
    save_path = os.path.join(plot_dir, filename + '_inertia.json')
=======
    save_path = os.path.join(plot_dir, 'inertia.json')
>>>>>>> 7a5aac2dd9d4f9c69e72f8b8068622d6c141fa00
    if os.path.isfile(save_path):
        with open(save_path, 'r') as fp:
            data = json.load(fp)
        inertia_dict.update(data)

    with open(save_path, 'w') as fp:
        json.dump(inertia_dict, fp)

<<<<<<< HEAD
def plot_silhouette(range_n_clusters, X, plot_dir, filename, kmeans_dict=None):

=======
def plot_silhouette(range_n_clusters, X, kmeans_dict=None, plot_dir=PLOT_DIR):
    #X = np_corpus_embeddings_trump
    # range_n_clusters = [5, 10]
>>>>>>> 7a5aac2dd9d4f9c69e72f8b8068622d6c141fa00
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    inertia_dict = {}

    start_time = time.time()
    for n_clusters in range_n_clusters:
        print('working on k means with k equals %s' % n_clusters)
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        if kmeans_dict is None:
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        else:
            clusterer = kmeans_dict[n_clusters]
            
        cluster_labels = clusterer.fit_predict(X)
        inertia_dict[n_clusters] = clusterer.inertia_

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        
<<<<<<< HEAD
        plt.savefig(os.path.join(plot_dir, filename + '_%s.jpg' % n_clusters), bbox_inches='tight')
=======
        plt.savefig(os.path.join(plot_dir, '%s.jpg' % n_clusters))
>>>>>>> 7a5aac2dd9d4f9c69e72f8b8068622d6c141fa00
        plt.close()
        print('time used for iteration with %s clusters is %s' % (n_clusters, time.time()-start_time))
        start_time = time.time()

    plt.close()

    return inertia_dict

def transform_embedding(embeddings, method: str, n_components:int=50):
    if method == 'raw':
        return embeddings
    elif method == 'pca':
        X_std = StandardScaler().fit_transform(embeddings)
    
        pca_model = PCA(n_components=n_components, random_state=random_state)
        coords = pca_model.fit_transform(X_std)
        return coords
    else:
        raise NotImplementedError('method not recognized: %s' % method)

# reference
## https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
## https://web.stanford.edu/~hastie/Papers/gap.pdf
## http://www.ee.columbia.edu/~dpwe/papers/PhamDN05-kmeans.pdf
## http://www.homepages.ucl.ac.uk/~ucakche/presentations/cqualitybolognahennig.pdf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('date', 
                        help='what values of k to try (int separated by comma, no space in-between')
    parser.add_argument('--list_of_k', default='[3,4,5,7,10,20,30,50]',type=str,
                        help='what values of k to try (int separated by comma, no space in-between')
    parser.add_argument('--topic', default='covid', type=str,
                        help='topic to decide figure storage folder')
    parser.add_argument('--subset_mode', default='string_match', type=str,
                        help='which subset: string_match or topic_model')
    parser.add_argument('--keyword_normed', default=False, type=bool,
                        help='embedding run: with or without the keywords')
    parser.add_argument('--method', default='raw', type=str,
                        help='data transformation before kmeans: choose from [raw, pca]')
    parser.add_argument('--dims', default=50, type=int,
                        help='number of dimensions for PCA preprocessing before kmeans')

    
<<<<<<< HEAD
    args = parser.parse_args()  
    range_n_clusters = _check_int(args.list_of_k[1:-1].split(','))
    
    data_path = os.path.join(DATA_PATH, args.topic)
    plot_dir = os.path.join(PLOT_DIR, args.topic)
    
    tweet_file = os.path.join(data_path, args.date+'_from_'+args.subset_mode+'.tsv')
    if args.keyword_normed!=True:
        embedding_file = os.path.join(data_path, args.date+'_from_'+args.subset_mode+'_embeddings.npy')
    else:
        embedding_file = os.path.join(data_path, args.date+'_from_'+args.subset_mode+'_keyword_normed_embeddings.npy')
    
    df, emb = read_in_data(tweet_file, embedding_file)
    emb = transform_embedding(emb, args.method, n_components=args.dims)
    
    filename = args.date+"_from_"+args.subset_mode
    if args.method != 'raw':
        filename += "_"+args.method+"_"+str(args.dims)
    if args.keyword_normed == True:
        filename += "_keyword_normed"

    inertia_dict = plot_silhouette(range_n_clusters, emb, plot_dir, filename)
    plot_kmeans_inertia(inertia_dict, plot_dir, filename)
    save_inertia(inertia_dict, plot_dir, filename)
    
=======
    args = parser.parse_args()    
    range_n_clusters = _check_int(args.list_of_k.split(','))
    df, np_corpus_embeddings_trump = read_in_data(embedding_file=embedding_file, tweet_file=tweet_file)

    embeddings = transform_embedding(np_corpus_embeddings_trump, args.method, n_components=args.dims)
    if args.method == 'pca':
        plot_dir = os.path.join(args.figure_folder, args.topic, args.method, str(args.dims))
    else:
        plot_dir = os.path.join(args.figure_folder, args.topic, args.method)

    inertia_dict = plot_silhouette(range_n_clusters, embeddings, plot_dir=plot_dir)
    plot_kmeans_inertia(inertia_dict, plot_dir=plot_dir)
    save_inertia(inertia_dict, plot_dir=plot_dir)
>>>>>>> 7a5aac2dd9d4f9c69e72f8b8068622d6c141fa00
