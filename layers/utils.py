import gzip
import cPickle as pickle
import numpy as np
import h5py
import theano
import theano.tensor as T


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


def norm_weight(nin, nout=None, scale=0.01, ortho=True, name=None):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    init_w = W.astype(theano.config.floatX)
    return theano.shared(init_w, name=name, borrow=True)


def zero_bias(nin, name):
    init_b = np.zeros(nin, dtype=theano.config.floatX)
    return theano.shared(init_b, name=name, borrow=True)
