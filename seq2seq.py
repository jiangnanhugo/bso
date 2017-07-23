import theano
import numpy as np
import theano.tensor as T
from recurrent import *
from softmax import *
from rnnblock import RnnBlock

if theano.config.device == 'cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams

from updates import *
from utils import norm_weight

import logging

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class Seq2Seq(object):
    def __init__(self, envocab_size, n_hidden, devocab_size, en_cell='bilstm',
                 de_cell='attlstm', optimizer='adam',
                 dropout=0.1):
        self.x = T.imatrix('batch sequence x')
        self.x_mask = T.fmatrix('batch sequence x mask')
        self.y = T.imatrix('batch sequence y')
        self.y_mask = T.fmatrix('batch sequence y mask')

        self.envocab_size = envocab_size
        self.n_hidden = n_hidden
        self.devocab_size = devocab_size

        self.en_cell = en_cell
        self.de_cell = de_cell
        self.optimizer = optimizer
        self.dropout = dropout

        self.en_loopup_table = theano.shared(name='Encoder look-up table',
                                             value=norm_weight(envocab_size, n_hidden),
                                             borrow=True)
        self.de_loopup_table = theano.shared(name='Decoder look-up table',
                                             value=norm_weight(devocab_size, n_hidden),
                                             borrow=True)
        self.is_train = T.iscalar('is_train')

        self.rng = RandomStreams(1234)
        self.build_graph()

    def build_graph(self):
        logger.debug('build rnn cell')
        embd_x = self.en_loopup_table[self.x]
        embd_y = self.de_loopup_table[self.y]
        y_shifted = T.zeros_like(embd_y)
        embd_y_shifted = T.set_subtensor(y_shifted[1:], embd_y[:-1])
        embd_y = embd_y_shifted

        # encoder: bidrection RNN
        # word embedding for forward rnn (source)
        encoder = None
        if self.en_cell == 'bilstm':
            encoder = BiLSTM(self.rng,self.envocab_size, self.n_hidden,
                             embd_x, self.x_mask,
                             output_mode='sum', is_train=self.is_train, dropout=self.dropout)
        elif self.en_cell == 'bigru':
            encoder = BiGRU(self.rng, self.envocab_size, self.n_hidden,
                            embd_x, self.x_mask,
                            output_mode='sum', is_train=self.is_train, dropout=self.dropout)
        elif self.en_cell.startswith('rnnblock'):
            mode = self.en_cell.split('.')[-1]
            print mode
            encoder = RnnBlock(self.rng, self.n_hidden,
                               embd_x, self.x_mask,
                               self.is_train, self.dropout, mode=mode)

        ctx = encoder.context
        # decoder init state
        init_state = encoder.activation
        decoder = None
        if self.de_cell == 'attlstm':
            decoder = CondLSTM(self.rng,
                              self.devocab_size, self.n_hidden,
                              embd_y, self.y_mask,
                              init_state=init_state, context=ctx, context_mask=self.x_mask,
                              is_train=self.is_train, dropout=self.dropout)
        # elif self.de_cell=='attgru':
        #    decoder=AttGRU(self.rng,
        #                self.devocab_size,self.n_hidden,
        #                embd_y,self.y_mask,
        #                init_state,context,self.is_train,self.dropout)

        output_layer = softmax(self.n_hidden, self.devocab_size, decoder.activation)
        cost = self.categorical_crossentropy(output_layer.activation, self.y, self.y_mask)
        self.params = [encoder.params, decoder.params]

        lr = T.scalar('lr')
        gparams = [T.clip(T.grad(cost, p) - 3, 3) for p in self.params]
        updates = None
        if self.optimizer == 'sgd':
            updates = sgd(self.params, gparams, lr)
        elif self.optimizer == 'adam':
            updates = adam(self.params, gparams, lr)
        elif self.optimizer == 'rmsprop':
            updates = rmsprop(self.params, gparams, lr)

        self.train = theano.function(inputs=[self.x, self.x_mask, self.y, self.y_mask, lr],
                                     outputs=cost,
                                     updates=updates,
                                     givens={self.is_train: np.cast['int32'](1)})

    def categorical_crossentropy(self, y_pred, y_true, y_mask):
        y_true = y_true.flatten()
        mask = y_mask.flatten()
        nll = T.nnet.categorical_crossentropy(y_pred, y_true)
        batch_nll = T.sum(nll * mask)
        return batch_nll / T.sum(mask)
