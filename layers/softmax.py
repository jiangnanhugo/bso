import numpy as np
import theano
import theano.tensor as T
from utils import *


class softmax(object):
    def __init__(self, n_input, n_output, x):
        self.n_input = n_input
        self.n_output = n_output

        self.x = x.reshape([-1, x.shape[-1]])

        init_W = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_input, n_output)), dtype=theano.config.floatX)
        init_b = np.zeros((n_output), dtype=theano.config.floatX)

        self.W = theano.shared(value=init_W, name='output_W', borrow=True)
        self.b = theano.shared(value=init_b, name='output_b', borrow=True)

        self.params = [self.W, self.b]

        self.activation = T.nnet.softmax(T.dot(self.x, self.W) + self.b)
        self.predict = T.argmax(self.activation, axis=-1)
        self.topk=T.argsort(self.activation,axis=-1)


class comb_softmax(object):
    def __init__(self, n_input, n_output, x, z,dimctx=None):
        self.n_input = n_input
        self.n_output = n_output
        if dimctx==None:
            dimctx=self.n_input

        self.x = x.reshape([-1, x.shape[-1]])
        self.z = z.reshape([-1, z.shape[-1]])

        self.W = norm_weight(n_input, n_output, name='output_W')
        self.U = norm_weight(dimctx, n_output, name='output_U')
        self.b = zero_bias(n_output, name='output_b')

        self.params = [self.W, self.b]
        logits = T.dot(self.x, self.W) +T.dot(self.z,self.U) + self.b

        self.activation = T.nnet.softmax(logits)
        self.predict = T.argmax(self.activation, axis=-1)
        self.topk = T.argsort(self.activation, axis=-1)

