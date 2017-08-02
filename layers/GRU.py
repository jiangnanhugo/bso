import numpy as np
import theano
import theano.tensor as T
from utils import *
from softmax import *

class GRU(object):
    def __init__(self, rng, n_input, n_hidden,
                 x, mask, init_state=None,
                 is_train=1, dropout=0.5, prefix="GRU_"):

        # https://github.com/nyu-dl/dl4mt-tutorial/tree/master/session3
        self.rng = rng
        prefix = prefix

        self.n_input = n_input
        self.n_hidden = n_hidden

        self.x = x
        self.mask = mask

        self.init_state = init_state
        self.is_train = is_train
        self.dropout = dropout

        # Update gate
        init_W = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_input, n_hidden * 2)),
                            dtype=theano.config.floatX)
        init_U = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_hidden, n_hidden * 2)),
                            dtype=theano.config.floatX)

        init_b = np.zeros((n_hidden * 2), dtype=theano.config.floatX)

        self.W = theano.shared(value=init_W, name=prefix + 'W')
        self.U = theano.shared(value=init_U, name=prefix + 'U')
        self.b = theano.shared(value=init_b, name=prefix + 'b')

        # Cell update
        init_Wx = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                               high=np.sqrt(1. / n_input),
                                               size=(n_input, n_hidden)),
                             dtype=theano.config.floatX)
        init_Ux = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                               high=np.sqrt(1. / n_input),
                                               size=(n_hidden, n_hidden)),
                             dtype=theano.config.floatX)
        init_bx = np.zeros((n_hidden), dtype=theano.config.floatX)

        self.Wx = theano.shared(value=init_Wx, name=prefix + 'Wx')
        self.Ux = theano.shared(value=init_Ux, name=prefix + 'Ux')
        self.bx = theano.shared(value=init_bx, name=prefix + 'bx')

        # Params
        self.params = [self.W, self.U, self.b, self.Wx, self.Ux, self.bx]

        self.build()

    def build(self):
        if self.init_state == None:
            self.init_state = T.zeros((self.x.shape[1], self.n_hidden), dtype=theano.config.floatX)
        state_below = T.dot(self.x, self.W) + self.b
        state_belowx = T.dot(self.x, self.Wx) + self.bx

        def split(x, n, dim):
            if x.ndim == 3:
                return x[:, :, n * dim: (n + 1) * dim]
            return x[:, n * dim:(n + 1) * dim]

        def _recurrence(x_t, xx_t, m, h_tm1):
            preact = T.dot(h_tm1, self.U)
            preact += x_t
            # reset fate
            r_t = T.nnet.sigmoid(split(preact, 0, self.n_hidden))
            # Update gate
            z_t = T.nnet.sigmoid(split(preact, 1, self.n_hidden))
            # Cell update
            c_t = T.tanh(T.dot(h_tm1, self.Ux) * r_t + xx_t)
            # Hidden state
            h_t = (1. - z_t) * c_t + z_t * h_tm1
            # masking
            h_t = h_t * m[:, None]
            return h_t

        h, _ = theano.scan(fn=_recurrence,
                           sequences=[state_below, state_belowx, self.mask],
                           outputs_info=self.init_state)
        self.context = h

        # Dropout
        if self.dropout > 0:
            drop_mask = self.rng.binomial(n=1, p=1 - self.dropout, size=h[-1].shape, dtype=theano.config.floatX)
            self.activation = T.switch(self.is_train, h[-1] * drop_mask, h[-1] * (1 - self.dropout))
        else:
            self.activation = T.switch(self.is_train, h[-1], h[-1])


class AttGRU(object):
    def __init__(self, rng, n_input, n_hidden,
                 x, mask=None, init_state=None, context=None, context_mask=None,
                 is_train=1, dropout=0.5,
                 one_step=False,dimctx=None):
        assert context, "Context must be provided"
        if one_step:
            assert init_state,'previous state must be provided.'
        # projected context
        assert context.ndim == 3, \
            'Context must be 3-d: [batch_size, timesteps, hidden_size*2]'



        prefix = "AttGRU_"

        # https://github.com/nyu-dl/dl4mt-tutorial/tree/master/session3
        self.rng = rng

        self.n_input = n_input
        self.n_hidden = n_hidden

        self.x = x
        self.mask = mask

        self.init_state = init_state
        self.context = context
        self.context_mask = context_mask
        self.is_train = is_train
        self.one_step=one_step
        self.dropout = dropout

        if dimctx is None:
            dimctx = n_hidden

        # layer 1
        # Update gate
        self.W = norm_weight(n_input, n_hidden * 2, name=prefix + 'W')
        self.U = norm_weight(n_hidden, n_hidden * 2, name=prefix + 'U')
        self.b = zero_bias(n_hidden * 2, prefix + 'b')
        # Cell update
        self.Wx = norm_weight(n_input, n_hidden, name=prefix + 'Wx')
        self.Ux = norm_weight(n_hidden, n_hidden, name=prefix + 'Ux')
        self.bx = zero_bias(n_hidden, prefix + 'bx')

        # layer 2
        # Update gate
        self.U_nl = norm_weight(n_hidden, n_hidden * 2, name='U_nl')
        self.b_nl = zero_bias(n_hidden * 2, 'b_nl')
        # Cell update
        self.Ux_nl = norm_weight(n_hidden, n_hidden, name='Ux_nl')
        self.bx_nl = zero_bias(n_hidden, 'bx_nl')

        # context to lstm
        self.Wc = norm_weight(dimctx, n_hidden * 2, name='Wc')
        self.Wcx = norm_weight(dimctx, n_hidden, name='Wcx')
        # attention
        # context -> hidden
        self.W_comb_att = norm_weight(n_hidden, dimctx, name='W_comb_att')
        self.Wc_att = norm_weight(dimctx, name='Wc_att')
        self.U_att = norm_weight(dimctx, 1, name='U_att')
        self.b_att = zero_bias((dimctx,), name='b_att')  # hidden bias
        self.c_tt = zero_bias((1,), 'c_tt')

        # Params
        self.params = [self.W, self.U, self.b, self.Wx, self.Ux, self.bx,
                       self.U_nl, self.b_nl, self.Ux_nl, self.bx_nl,
                       self.Wc, self.Wcx, self.W_comb_att,
                       self.Wc_att, self.U_att, self.b_att, self.c_tt]

        self.build()

    def build(self):

        def split(x, n, dim):
            if x.ndim == 3:
                return x[:, :, n * dim: (n + 1) * dim]
            return x[:, n * dim:(n + 1) * dim]

        def _recurrence(x_t, xx_t, m,
                        h_, ctx_,
                        pctx_, cc_):
            # pctx: 27 20 600
            # cc_ 27 20 600
            preact1 = T.nnet.sigmoid(x_t + T.dot(h_, self.U))

            r1 = split(preact1, 0, self.n_hidden)
            u1 = split(preact1, 1, self.n_hidden)
            h1 = T.tanh(T.dot(h_, self.Ux) * r1 + xx_t)
            h1 = u1 * h_ + (1. - u1) * h1
            h1 = m[:, None] * h1 + (1. - m)[:, None] * h_

            # attention

            # pstate_: 20 600
            pstate_ = T.dot(h1, self.W_comb_att)
            pctx__ = pctx_ + pstate_[None, :, :]
            pctx__ = T.tanh(pctx__)
            # pctx__: 27 20 600
            # alpha: 27 20 1
            alpha = T.dot(pctx__, self.U_att) + self.c_tt
            alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
            # alpha 27 20
            alpha = T.exp(alpha)
            # if self.context_mask:
            alpha = alpha * self.context_mask
            alpha = alpha / alpha.sum(0, keepdims=True)
            # alpha 27 20
            ctx_ = T.sum(cc_ * alpha[:, :, None], axis=0)  # batch_size, dimctx
            # ctx_ 20 600

            # another gru cell
            # ctx_:[batch_size,dimctx],
            # h1: [batch_size, hidden_size]
            # preact2 = T.nnet.sigmoid(T.dot(h1, self.U_nl) + T.dot(ctx_, self.Wc) + self.b_nl)
            preact2 = T.dot(h1, self.U_nl) + self.b_nl
            preact2 += T.dot(ctx_, self.Wc)
            preact2 = T.nnet.sigmoid(preact2)

            r2 = split(preact2, 0, self.n_hidden)
            u2 = split(preact2, 1, self.n_hidden)
            # _t2 = T.tanh((T.dot(h1, self.Ux_nl) + self.bx_nl) * r2 + T.dot(ctx_, self.Wcx))
            preactx2 = T.dot(h1, self.Ux_nl) + self.bx_nl
            preactx2 *= r2
            preactx2 += T.dot(ctx_, self.Wcx)
            h2 = T.tanh(preactx2)
            h2 = u2 * h1 + (1. - u2) * h2
            h2 = m[:, None] * h2 + (1. - m)[:, None] * h1
            return h2, ctx_

        if self.init_state == None:
            self.init_state = T.zeros((self.x.shape[1], self.n_hidden), dtype=theano.config.floatX)

        # projected x
        state_below = T.dot(self.x, self.W) + self.b
        state_belowx = T.dot(self.x, self.Wx) + self.bx
        # mask
        print self.mask
        if self.mask == None:
            self.mask = T.alloc(1., state_below.shape[0],1)

        # pctx_:[seqlen,batch_size,dimctx]
        # self.context:[seqlen,batch_size,dimctx]:
        # self.context:[27,20,600]
        # pctx: [27,20,600]
        pctx_ = T.dot(self.context, self.Wc_att) + self.b_att
        seqs=[state_below, state_belowx, self.mask]
        if self.one_step==True:
            hidden_states,contexts=_recurrence(*(seqs+[self.init_state,None,pctx_,self.context]))
        else:
            [hidden_states, contexts], _ = theano.scan(fn=_recurrence,
                                                   sequences=seqs,
                                                   non_sequences=[pctx_, self.context],
                                                   outputs_info=[self.init_state,
                                                                 T.alloc(0., self.context.shape[1],
                                                                         self.context.shape[2])])
        self.hidden_states = hidden_states
        self.contexts = contexts

        # Dropout
        if self.dropout > 0:
            drop_mask = self.rng.binomial(n=1, p=1 - self.dropout, size=hidden_states.shape, dtype=theano.config.floatX)
            self.hidden_states = T.switch(self.is_train, hidden_states * drop_mask, hidden_states * (1 - self.dropout))
        else:
            self.activation = T.switch(self.is_train, hidden_states, hidden_states)
        # Dropout
        if self.dropout > 0:
            drop_mask = self.rng.binomial(n=1, p=1 - self.dropout, size=contexts.shape,
                                          dtype=theano.config.floatX)
            self.contexts = T.switch(self.is_train, contexts * drop_mask,
                                     contexts * (1 - self.dropout))
        else:
            self.contexts = T.switch(self.is_train, contexts, contexts)



class BiGRU(object):
    def __init__(self, rng, n_input, n_hidden,
                 x, mask,
                 is_train=1, dropout=0.5):
        # https://github.com/nyu-dl/dl4mt-tutorial/tree/master/session3
        prefix = "BiGRU_"
        self.rng = rng

        self.n_input = n_input
        self.n_hidden = n_hidden

        self.x = x
        self.mask = mask

        self.is_train = is_train
        self.dropout = dropout

        # Update gate
        self.W = norm_weight(n_input, n_hidden * 2, name=prefix + 'W')
        self.U = norm_weight(n_hidden, n_hidden * 2, name=prefix + 'U')
        self.b = zero_bias(n_hidden * 2, prefix + 'b')

        # Cell update
        self.Wx = norm_weight(n_input, n_hidden, name=prefix + 'Wx')
        self.Ux = norm_weight(n_hidden, n_hidden, name=prefix + 'Ux')
        self.bx = zero_bias(n_hidden, prefix + 'bxr')

        # backword
        # Update gate
        self.Wr = norm_weight(n_input, n_hidden * 2, name=prefix + 'Wr')
        self.Ur = norm_weight(n_hidden, n_hidden * 2, name=prefix + 'Ur')
        self.br = zero_bias(n_hidden * 2, prefix + 'br')

        # Cell update
        self.Wxr = norm_weight(n_input, n_hidden, name=prefix + 'Wxr')
        self.Uxr = norm_weight(n_hidden, n_hidden, name=prefix + 'Uxr')
        self.bxr = zero_bias(n_hidden, prefix + 'bxr')
        # context vector
        self.Wc = norm_weight(n_hidden * 2, n_hidden, name=prefix + 'Wc')
        self.bc = zero_bias(n_hidden, prefix + 'bc')

        # Params
        self.params = [self.W, self.U, self.b, self.Wx, self.Ux, self.bx,
                       self.Wr, self.Ur, self.br, self.Wxr, self.Uxr, self.bxr,
                       self.Wc, self.bc]

        self.build_graph()

    def build_graph(self):
        def split(x, n, dim):
            if x.ndim == 3:
                return x[:, :, n * dim: (n + 1) * dim]
            return x[:, n * dim:(n + 1) * dim]

        def _recurrence(x_t, xx_t, m, h_tm1, U, Ux):
            preact = x_t + T.dot(h_tm1, U)
            # reset fate
            r_t = T.nnet.sigmoid(split(preact, 0, self.n_hidden))
            # Update gate
            z_t = T.nnet.sigmoid(split(preact, 1, self.n_hidden))
            # Cell update
            c_t = T.tanh(T.dot(h_tm1, Ux) * r_t + xx_t)
            # Hidden state
            h_t = (1. - z_t) * c_t + z_t * h_tm1
            # masking
            h_t = h_t * m[:, None]
            return h_t

        state_pre = T.zeros((self.x.shape[1], self.n_hidden), dtype=theano.config.floatX)
        state_below = T.dot(self.x, self.W) + self.b
        state_belowx = T.dot(self.x, self.Wx) + self.bx
        h, _ = theano.scan(fn=_recurrence,
                           sequences=[state_below, state_belowx, self.mask],
                           non_sequences=[self.U, self.Ux],
                           outputs_info=state_pre)

        state_pre = T.zeros((self.x.shape[1], self.n_hidden), dtype=theano.config.floatX)
        state_below = T.dot(self.x, self.Wr) + self.br
        state_belowx = T.dot(self.x, self.Wxr) + self.bxr
        hr, _ = theano.scan(fn=_recurrence,
                            sequences=[state_below, state_belowx, self.mask],
                            non_sequences=[self.Ur, self.Uxr],
                            go_backwards=True,
                            outputs_info=state_pre)

        # context will be the concatenation of forward and backward rnns
        ctx = T.concatenate([h, hr], axis=- 1)
        self.context = ctx

        # mean of the context (across time) will be used to initialize decoder rnn
        ctx_mean = (ctx * self.mask[:, :, None]).sum(0) / self.mask.sum(0)[:, None]
        # or you can use the last state of forward + backward encoder rnns
        # ctx_last = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)


        self.activation = T.tanh(T.dot(ctx_mean, self.Wc) + self.bc)
