import argparse

import numpy as np


gender_vector = '/Users/boyuliu/Dropbox (MIT)/nlp_project/data/gender_bias/trump_tweets_gender_pca_vector.npy'

def project_embed(embed, vec, onto=False):
    assert vec.shape[0] == embed.shape[1]
    vec = vec.reshape(1,-1)
    scale = np.einsum('ij,ij->i', embed, vec).reshape(-1, 1)
    projections = np.matmul(scale, vec)
    if onto:
        return projections
    else:
        transformed_embedding = embed - projections
        return transformed_embedding

def process_embedding(embed_file, vec, onto=False, vec_name='gender'):
    embed_name = embed_file.split('/')[-1]
    embed_folder = embed_file.strip(embed_name)

    embeddings = np.load(embed_file)
    transformed_embedding = project_embed(embeddings, vec, onto=onto)

    postfix = '_%s_%s.npy' % ('onto' if onto else 'remove', vec_name)
    new_file_name = embed_name.strip('.npy') + postfix
    np.save(embed_folder+new_file_name, transformed_embedding)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('embed_file', 
                        help='path to topic_embedding file')
    parser.add_argument('--vec', default=gender_vector, type=str,
                        help='vector representing dimension you are trying to remove or project onto')
    parser.add_argument('--onto', default=False, type=bool,
                        help='whether to project onto the direction or remove that direction')
    parser.add_argument('--topic', default='gender', type=str,
                        help='topic to decide new embeddings file name')
    args = parser.parse_args()    

    
    vector = np.load(args.vec)

    # trump_folder = '/Users/boyuliu/Dropbox (MIT)/nlp_project/data/topical_tweets/trump/str_match'
    # embed_file = '/20200101_from_string_match_embeddings.npy'
    # embeddings = np.load(trump_folder+embed_file)

    # trump_folder = '/Users/boyuliu/Dropbox (MIT)/nlp_project/data/topic_embeddings/'
    # embed_file = '/trump_embeddings.npy'
    process_embedding(args.embed_file, vector, onto=args.onto, vec_name=args.topic)
