import numpy as np
import scipy.io
from sklearn.decomposition import PCA
from sklearn import preprocessing

#data =  scipy.io.mmread("")


def format_func(data):
    data = data.toarray()
    return data.transpose()

def normalize(data):
    data = data - data.min()
    return (data/data.max() *2)-1

def data_format(data):
    data = format_func(data)
    return normalize(data)



def pca_func(data, desired_dimensions):
    data = preprocessing.scale(data).transpose()
    pca = PCA(n_components = desired_dimensions)
    #pca.fit(data)
    return pca.fit_transform(data).transpose(), pca
