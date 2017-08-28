import theano
import numpy as np
import theano.tensor as T
from layers.recurrent import *
from layers.softmax import *

if theano.config.device == 'cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    # from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from updates import *
from layers.utils import norm_weight

import logging

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class Seq2Seq(object):
    def __init__(self, envocab_size, devocab_size, n_hidden, rnn_cells=None,
                 optimizer='adam', dropout=0.1, one_step=False):

        self.enc = T.imatrix('encoder input')
        self.enc_mask = T.fmatrix('enc mask')
        self.envocab_size = envocab_size
        self.n_hidden = n_hidden
        self.devocab_size = devocab_size

        self.en_cell, self.de_cell = rnn_cells

        self.optimizer = optimizer
        self.dropout = dropout

        self.one_step = one_step

        self.en_loopup_table = norm_weight(envocab_size, n_hidden, name='Encoder look-up table')
        self.de_loopup_table = norm_weight(devocab_size, n_hidden, name='Decoder look-up table')

        self.is_train = T.iscalar('is_train')
        self.srng = RandomStreams(1234)
        self.eps = 1e-8
        self.build_enc()
        if self.one_step == False:
            self.build_dec()
        else:
            self.build_sample()

    def build_enc(self):
        enc_shape = self.enc.shape
        embd_enc = self.en_loopup_table[self.enc.flatten()]
        embd_enc = embd_enc.reshape([enc_shape[0], enc_shape[1], -1])

        # encoder: bidrection RNN
        # word embedding for forward rnn (source)
        logger.debug('calculating bidirectional encoder.....')

        if self.en_cell == 'bilstm':
            self.encoder = BiLSTM(self.srng, self.n_hidden, self.n_hidden,
                                  embd_enc, self.enc_mask,
                                  output_mode='sum', is_train=self.is_train, dropout=self.dropout)
        elif self.en_cell == 'bigru':
            self.encoder = BiGRU(self.srng, self.n_hidden, self.n_hidden,
                                 embd_enc, self.enc_mask,
                                 is_train=self.is_train, dropout=self.dropout)
        elif self.en_cell == 'gru':
            self.encoder = GRU(self.srng, self.n_hidden, self.n_hidden,
                               embd_enc, self.enc_mask,
                               is_train=self.is_train, dropout=self.dropout)
        elif self.en_cell.startswith('rnnblock'):
            mode = self.en_cell.split('.')[-1]
            self.encoder = RnnBlock(self.srng, self.n_hidden,
                                    embd_enc, self.enc_mask,
                                    self.is_train, self.dropout, mode=mode)

    def build_dec(self):
        dec_input = T.imatrix('decoder input')
        dec_output = T.imatrix('decoder output')
        dec_mask = T.fmatrix('decoder mask')
        ss_prob=T.fscalar("schedule sampling probability")
        logger.debug('build decoder rnn cell....')

        dec_shape = dec_input.shape
        embd_dec_input = self.de_loopup_table[dec_input.flatten()]
        embd_dec_input = embd_dec_input.reshape([dec_shape[0], dec_shape[1], -1])
        # decoder init state
        init_state = self.encoder.activation
        ctx = self.encoder.context
        logger.debug('calculating decoder with attention model')
        decoder = None
        if self.de_cell == 'attlstm':
            decoder = AttLSTM(self.srng,
                              self.n_hidden, self.n_hidden,
                              embd_dec_input, dec_mask,
                              init_state=init_state, context=ctx, context_mask=self.enc_mask,
                              is_train=self.is_train, dropout=self.dropout)
        elif self.de_cell == 'attgru':
            decoder = AttGRU(self.srng,
                             self.n_hidden, self.n_hidden,
                             embd_dec_input, dec_mask,
                             init_state=init_state, context=ctx, context_mask=self.enc_mask,
                             is_train=self.is_train, dropout=self.dropout, dimctx=self.n_hidden * 2)
        elif self.de_cell == 'ssattgru':
            decoder = SSAttGRU(self.srng,
                               self.n_hidden, self.n_hidden, self.devocab_size,
                               dec_input, dec_mask, self.de_loopup_table,
                               init_state=init_state, context=ctx, context_mask=self.enc_mask,
                               is_train=self.is_train, dimctx=self.n_hidden * 2, ss_prob=ss_prob)

        self.params = [self.en_loopup_table, self.de_loopup_table]
        self.params += self.encoder.params
        self.params += decoder.params

        if self.de_cell == 'ssattgru':
            cost, acc = self.categorical_crossentropy(decoder, dec_output, dec_mask)
        else:
            # hidden states of the decoder gru
            # target_seq_len, batch_size, hidden_size: (19, 20, 300)
            hidden_states = decoder.hidden_states
            # weighted averages of context, generated by attention module
            contexts = decoder.contexts
            # weights (alignment matrix)
            # alignment_matries = decoder.activation[2]
            output_layer = comb_softmax(self.n_hidden, self.devocab_size, hidden_states, contexts,
                                        dimctx=self.n_hidden * 2)
            self.params += output_layer.params
            # output_layer = softmax(self.n_hidden, self.devocab_size, hidden_states)
            cost, acc = self.categorical_crossentropy(output_layer, dec_output, dec_mask)

        logger.debug('calculating gradient update')

        lr = T.scalar('lr')
        gparams = [T.clip(T.grad(cost, p), - 3., 3.) for p in self.params]
        updates = None
        if self.optimizer == 'sgd':
            updates = sgd(self.params, gparams, lr)
        elif self.optimizer == 'adam':
            updates = adam(self.params, gparams, lr)
        elif self.optimizer == 'rmsprop':
            updates = rmsprop(self.params, gparams, lr)

        logger.debug('compling final function......')
        input_list=[self.enc, self.enc_mask,dec_input, dec_output, dec_mask, lr]
        if self.de_cell=='ssattgru':
            input_list.append(ss_prob)
        self.train = theano.function(inputs=input_list,
                                     outputs=[cost, acc],
                                     updates=updates)
                                     #givens={self.is_train: np.cast['int32'](1)})

    def build_sample(self):
        logger.info("build sample.....")
        dec_input = T.ivector('decoder input')
        # embd_dec_input = self.de_loopup_table[dec_input]
        embd_dec_input = T.switch(dec_input[:, None] < 0,
                                  T.alloc(0., 1, self.n_hidden),
                                  self.de_loopup_table[dec_input])
        self.encoder_state = theano.function(inputs=[self.enc, self.enc_mask],
                                             outputs=[self.encoder.activation, self.encoder.context],
                                             name='encoder outputs')

        init_state = T.fmatrix('decoder init_state')
        # batch_size, hidden_size
        decoder = AttGRU(self.srng,
                         self.n_hidden, self.n_hidden,
                         embd_dec_input, mask=None,
                         init_state=init_state, context=self.encoder.context, context_mask=None,  # self.enc_mask,
                         is_train=self.is_train, dropout=self.dropout, one_step=True, dimctx=self.n_hidden * 2)

        output_layer = softmax(self.n_hidden, self.devocab_size, decoder.hidden_states)
        next_prob = output_layer.activation

        self.params = [self.en_loopup_table, self.de_loopup_table]
        self.params += self.encoder.params
        self.params += decoder.params
        self.params += output_layer.params

        self.predict = theano.function(inputs=[dec_input, self.encoder.context, init_state],
                                       outputs=[next_prob, decoder.hidden_states],
                                       givens={self.is_train: np.cast['int32'](0)},
                                       name='next prob generator')

    def categorical_crossentropy(self, output, y_true, y_mask):
        y_prob = output.activation
        y_prob = y_prob.reshape((-1, y_prob.shape[-1]))
        y_pred = output.predict.flatten()
        mask = y_mask.flatten()
        y_true = y_true.flatten()
        nll = T.nnet.categorical_crossentropy(y_prob, y_true)
        batch_nll = T.sum(nll * mask)
        batch_acc = T.sum(T.eq(y_pred, y_true) * mask)
        return batch_nll / T.sum(mask), batch_acc / T.sum(mask)
