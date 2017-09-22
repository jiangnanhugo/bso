import theano
import numpy as np
import theano.tensor as T
from theano.gpuarray import dnn
from theano.gpuarray.type import gpuarray_shared_constructor
from utils import *



class RnnBlock(object):
    def __init__(self,rng,n_hidden,x,
                 xmask,is_train,dropout,mode='gru',
                 n_layer=1, pre_state=None,**kwargs):

        prefix = "BiGRU_"
        Wc = norm_weight(n_hidden * 2, n_hidden, name=prefix + 'Wc')
        bc = zero_bias(n_hidden, prefix + 'bc')

        self.is_train=is_train
        self.dropout=dropout

        self.rng=rng
        self.xmask=xmask

        if pre_state==None:
            h0 = T.zeros((n_layer, x.shape[1], n_hidden), dtype=theano.config.floatX)
            pre_state = [h0, ]
            if mode=='lstm':
                c0 = T.zeros((n_layer, x.shape[1], n_hidden), dtype=theano.config.floatX)
                pre_state.append(c0)


        rnnb=dnn.RNNBlock(dtype=theano.config.floatX,
                          hidden_size=n_hidden,
                          num_layers=n_layer,
                          rnn_mode=mode,
                          input_mode='skip',
                          direction_mode='bidirectional')
        psize=rnnb.get_param_size([1,n_hidden])
        print psize
        params_cudnn = gpuarray_shared_constructor(
            np.zeros((psize,), dtype=theano.config.floatX)
        )
        #l = np.sqrt(6.) / np.sqrt(4 * n_hidden)
        #pvalue = np.asarray(self.rng.uniform(low=-l, high=l, size=(psize,)), dtype=theano.config.floatX)
        #params_cudnn=gpuarray_shared_constructor(pvalue,name='cudnn')
        self.params=[params_cudnn,]

        if mode=='lstm':
            h=rnnb.apply(params_cudnn,x,pre_state[0],pre_state[1])[0]
        else:
            h=rnnb.apply(params_cudnn,x,pre_state[0])[0]

        h=h*self.xmask.dimshuffle(0,1,'x')
        self.context=h

        ctx_mean = (h * self.xmask[:, :, None]).sum(0) / self.xmask.sum(0)[:, None]

        self.activation = T.tanh(T.dot(ctx_mean, Wc) + bc)

        # Dropout
        if self.dropout > 0:
            drop_mask = self.rng.binomial(n=1, p=1 - self.dropout, size=h.shape, dtype=theano.config.floatX)
            self.activation = T.switch(self.is_train, h * drop_mask, h * (1 - self.dropout))
        else:
            self.activation = T.switch(self.is_train, h, h)



