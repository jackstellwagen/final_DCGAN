import numpy as np
import scipy.io
from sklearn.decomposition import PCA


#data =  scipy.io.mmread("")


def format(data):
    data = data.toarray()
    return data.transpose()

def normalize(data):
    data = data - data.min()
    return (data/data.max() *2)-1

def data_format(data):
    data = format(data)
    return normalize(data)

def pca_func(data, desired_dimensions):
    data = data.transpose()
    pca = PCA(n_components = desired_dimensions)
    pca.fit(data)
    return pca.transform(data).transpose(), pca
