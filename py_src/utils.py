import numpy as np
from scipy.io import loadmat

def logSpecDbDist(x,y):
    
    size = x.shape[0]
    assert y.shape[0] == size

    sumSqDiff = 0.0
    for k in range(len(x)):
        diff = x[k] - y[k]
        sumSqDiff += diff * diff

    dist = np.sqrt(sumSqDiff*2)*(10/np.log(10))
    return dist

def read_mcc(path):
    d = np.fromfile(path,dtype=np.float32)
    d = np.reshape(d, (40, d.size//40), order='F')
    d = np.transpose(d)

    return d

def read_mat(path):
    d = loadmat(path)
    return d
