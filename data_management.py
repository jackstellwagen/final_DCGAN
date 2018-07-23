import numpy as np
import scipy.io


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

