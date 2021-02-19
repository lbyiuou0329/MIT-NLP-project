import sys
# import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


random_state = 123

def pca_analysis(X, n_components=2, random_state=random_state):
    
    X_std = StandardScaler().fit_transform(X)
    
    pca_model = PCA(n_components=n_components, random_state=random_state)
    coords = pca_model.fit_transform(X_std)
    explained_variance = pca_model.explained_variance_ratio_
        
    return pca_model, coords, explained_variance

def tsne_analysis(X, n_components=2, random_state=random_state):
    
    tsne_model = TSNE(n_components=n_components, random_state=random_state)
    coords = tsne_model.fit_transform(X)
    
    return coords

def dim_reduc_plot(coords, color_var=None):
    
    plt.subplots(figsize=(10,10))
    plt.scatter(
        coords[:,0], coords[:,1], 
        c=color_var,
        marker='.'
    )
    
    plt.show()
    
def k_means(X, n_clusters, random_state=random_state):

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    C = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    
    return kmeans, labels, C, inertia

def plot_kmeans_inertia(inertia_dict):
    keys = sorted(inertia_dict.keys())
    values = [inertia_dict[k] for k in keys]
    plt.plot(keys, values)
    plt.show()