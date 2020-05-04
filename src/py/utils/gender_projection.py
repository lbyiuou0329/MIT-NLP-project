import os
import time
import argparse

import pandas as pd
import numpy as np


def remove_vec_from_embed(embed, vec):
	assert vec.shape[0] == embed.shape[1]
	vec = vec.reshape(1,-1)
	scale = np.einsum('ij,ij->i', embed, vec).reshape(-1, 1)
	projections = np.matmul(scale, vec)
	transformed_embedding = embed - projections
	return transformed_embedding

def process_embedding(embed_folder, embed_name, vec):
	embeddings = np.load(embed_folder+embed_name)
	transformed_embedding = remove_vec_from_embed(embeddings, vec)
	new_file_name = embed_name.strip('.npy') + '_remove_gender.npy'
	np.save(embed_folder+new_file_name, transformed_embedding)



if __name__ == '__main__':
	gender_vector = '/Users/boyuliu/Dropbox (MIT)/nlp_project/data/gender_bias/trump_tweets_gender_pca_vector.npy'
	gender_vector = np.load(gender_vector)

	# trump_folder = '/Users/boyuliu/Dropbox (MIT)/nlp_project/data/topical_tweets/trump/str_match'
	# embed_file = '/20200101_from_string_match_embeddings.npy'
	# embeddings = np.load(trump_folder+embed_file)

	trump_folder = '/Users/boyuliu/Dropbox (MIT)/nlp_project/data/topic_embeddings/'
	embed_file = '/trump_embeddings.npy'
	process_embedding(trump_folder, embed_file, gender_vector)

	
    # parser = argparse.ArgumentParser()
    # parser.add_argument('list_of_k', 
    #                     help='what values of k to try (int separated by comma, no space in-between')
    # parser.add_argument('--datapath', default=datapath,
    #                     help='path to folder with topic_embeddings')
    # parser.add_argument('--figure_folder', default=PLOT_DIR,
    #                     help='path to store figures')
    # parser.add_argument('--method', default='raw', type=str,
    #                     help='data transformation before kmeans')
    # parser.add_argument('--topic', default='trump', type=str,
    #                     help='topic to decide figure storage folder')
    # parser.add_argument('--dims', default=50, type=int,
    #                     help='number of dimensions for PCA preprocessing before kmeans')

    
    # args = parser.parse_args()    
    # range_n_clusters = _check_int(args.list_of_k.split(','))
    # df, np_corpus_embeddings_trump = read_in_data()

    # embeddings = transform_embedding(np_corpus_embeddings_trump, args.method, n_components=args.dims)
    # if args.method == 'pca':
    #     plot_dir = os.path.join(args.figure_folder, args.topic, args.method, str(args.dims))
    # else:
    #     plot_dir = os.path.join(args.figure_folder, args.topic, args.method)

    # inertia_dict = plot_silhouette(range_n_clusters, embeddings, PLOT_DIR=plot_dir)
    # plot_kmeans_inertia(inertia_dict, PLOT_DIR=plot_dir)
    # save_inertia(inertia_dict, PLOT_DIR=plot_dir)
