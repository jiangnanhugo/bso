import numpy as np
import theano
import theano.tensor as T
from utils import *


class GRU(object):
    def __init__(self, rng, n_input, n_hidden,
                 x, E, mask, init_state=None,
                 is_train=1, p=0.5):

        # https://github.com/nyu-dl/dl4mt-tutorial/tree/master/session3
        self.rng = rng

        self.n_input = n_input
        self.n_hidden = n_hidden

        self.x = x
        self.E = E
        self.mask = mask

        self.inti_state = init_state
        self.is_train = is_train
        self.p = p

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

        self.W = theano.shared(value=init_W, name='W')
        self.U = theano.shared(value=init_U, name='U')
        self.b = theano.shared(value=init_b, name='b')

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

        self.Wx = theano.shared(value=init_Wx, name='Wx')
        self.Ux = theano.shared(value=init_Ux, name='Ux')
        self.bx = theano.shared(value=init_bx, name='bx')

        # Params
        self.params = [self.W, self.U, self.b, self.Wx, self.Ux, self.bx]

        self.build()

    def build(self):
        if self.inti_state == None:
            self.init_state = T.zeros((self.x.shape[-1], self.n_hidden), dtype=theano.config.floatX)
        state_below = T.dot(self.E[self.x, :], self.W) + self.b
        state_belowx = T.dot(self.E[self.x, :], self.Wx) + self.bx

        def split(x, n, dim):
            if x.ndim == 3:
                return x[:, :, n * dim: (n + 1) * dim]
            return x[:, n * dim:(n + 1) * dim]

        def _recurrence(x_t, xx_t, m, h_tm1):
            preact = x_t + T.dot(h_tm1, self.U)

            # reset fate
            r_t = T.nnet.sigmoid(split(preact, 0, self.n_hidden))
            # Update gate
            z_t = T.nnet.sigmoid(split(preact, 1, self.n_hidden))

            # Cell update
            c_t = T.tanh(T.dot(h_tm1, self.Ux) * r_t + xx_t)

            # Hidden state
            h_t = (T.ones_like(z_t) - z_t) * c_t + z_t * h_tm1

            # masking
            h_t = h_t * m[:, None]

            return h_t

        h, _ = theano.scan(fn=_recurrence,
                           sequences=[state_below, state_belowx, self.mask],
                           outputs_info=self.init_state,
                           truncate_gradient=-1)

        # Dropout
        if self.p > 0:
            drop_mask = self.rng.binomial(n=1, p=1 - self.p, size=h.shape, dtype=theano.config.floatX)
            self.activation = T.switch(self.is_train, h * drop_mask, h * (1 - self.p))
        else:
            self.activation = T.switch(self.is_train, h, h)


class AttGRU(object):
    def __init__(self, rng, n_input, n_hidden,
                 x, mask, init_state=None, context=None, context_mask=None,
                 is_train=1, dropout=0.5,
                 dimctx=None):
        assert context, "Context must be provided"

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
        self.dropout = dropout

        if dimctx is None:
            dimctx = n_hidden

        # layer 1
        # Update gate
        self.W = norm_weight(n_input, n_hidden * 2, name='W')
        self.U = norm_weight(n_hidden, n_hidden * 2, name='U')
        self.b = zero_bias(n_hidden * 2, 'b')
        # Cell update
        self.Wx = norm_weight(n_input, n_hidden, name='Wx')
        self.Ux = norm_weight(n_hidden, n_hidden, name='Ux')
        self.bx = zero_bias(n_hidden, 'bx')

        # layer 2
        # Update gate
        self.U_nl = norm_weight(n_hidden, n_hidden * 2, name='U_nl')
        self.b_nl = zero_bias(n_hidden * 2, 'b_nl')
        # Cell update
        self.Ux_nl = norm_weight(n_hidden, n_hidden, name='Ux_nl')
        self.bx_nl = zero_bias(n_hidden, 'bx_nl')

        # context to lstm
        self.Wc = norm_weight(dimctx, n_hidden * 2, name='Wc')
        self.Wcx = norm_weight(dimctx, n_hidden * 2, name='Wcx')
        # attention
        # context -> hidden
        self.W_comb_att = norm_weight(n_hidden, dimctx, name='W_comb_att')
        self.Wc_att = norm_weight(dimctx, name='Wc_att')

        self.U_att = norm_weight(dimctx, 1, name='u_att')
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
                        h_tm1, ctx_t,  # alpha_t,
                        pctx_, cc_):
            preact = T.nnet.sigmoid(x_t + T.dot(h_tm1, self.U))

            # reset fate
            r_t = split(preact, 0, self.n_hidden)
            # Update gate
            z_t = split(preact, 1, self.n_hidden)
            # Cell update
            c_t = T.tanh(T.dot(h_tm1, self.Ux) * r_t + xx_t)
            # Hidden state
            h_t = z_t * h_tm1 + (1. - z_t) * c_t
            # masking
            h_t = h_t * m[:, None] + (1. - m)[:, None] * h_tm1

            # attention
            alpha = T.dot(T.tanh(T.dot(h_t, self.W_comb_att) + pctx_), self.U_att) + self.c_tt
            alpha = alpha.reshape([alpha.shape[0], -1])
            alpha = T.exp(alpha)
            if self.context_mask:
                alpha = alpha * self.context_mask
            alpha = alpha / alpha.sum(0, keepdims=True)
            ctx_ = (cc_ * alpha[:, :, None]).sum(0)

            # another gru cell
            preact2 = T.dot(h_t, self.U_nl) + T.dot(ctx_, self.Wc) + self.b_nl

            # reset fate # Update gate
            r_t2 = T.nnet.sigmoid(split(preact2, 0, self.n_hidden))
            z_t2 = T.nnet.sigmoid(split(preact2, 1, self.n_hidden))
            # Cell update
            c_t2 = T.tanh((T.dot(h_t, self.Ux_nl) + self.bx_nl) * r_t2 + T.dot(ctx_, self.Wcx))
            # Hidden state
            h_t2 = (1. - z_t2) * c_t2 + z_t2 * h_t
            # masking
            h_t2 = h_t2 * m[:, None] + (1. - m)[:, None] * h_t
            # layer2 gru,  # current context, alpha
            return h_t2, ctx_,  # alpha.T

        if self.init_state == None:
            self.init_state = T.zeros((self.x.shape[-1], self.n_hidden), dtype=theano.config.floatX)

        # projected context
        assert self.context.ndim == 2, \
            'Context must be 2-d: # annotation x #sample x dim'
        # projected x
        state_below = T.dot(self.x, self.W) + self.b
        state_belowx = T.dot(self.x, self.Wx) + self.bx
        pctx_ = T.dot(self.context, self.Wc_att) + self.b_att

        [hidden_states, contexts], _ = theano.scan(fn=_recurrence,
                                                   sequences=[state_below, state_belowx, self.mask],
                                                   non_sequences=[pctx_, self.context],
                                                   outputs_info=[self.init_state,
                                                                 T.zeros_like(self.context)],
                                                   # T.alloc(0., self.context.shape[0],self.context.shape[0])],
                                                   truncate_gradient=-1)

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
                 x, mask, output_mode='sum',
                 is_train=1, dropout=0.5):
        # https://github.com/nyu-dl/dl4mt-tutorial/tree/master/session3
        self.rng = rng

        self.n_input = n_input
        self.n_hidden = n_hidden

        self.x = x
        self.mask = mask

        self.output_mode = output_mode
        self.is_train = is_train
        self.dropout = dropout

        # Update gate
        self.W = norm_weight(n_input, n_hidden * 2, name='W')
        self.U = norm_weight(n_hidden, n_hidden * 2, name='U')
        self.b = zero_bias(n_hidden * 2, 'b')

        # Cell update
        self.Wx = norm_weight(n_input, n_hidden, name='Wx')
        self.Ux = norm_weight(n_hidden, n_hidden, name='Ux')
        self.bx = zero_bias(n_hidden, 'bx')

        # backword
        # Update gate
        self.Wr = norm_weight(n_input, n_hidden * 2, name='Wr')
        self.Ur = norm_weight(n_hidden, n_hidden * 2, name='Ur')
        self.br = zero_bias(n_hidden * 2, 'br')

        # Cell update
        self.Wxr = norm_weight(n_input, n_hidden, name='Wxr')
        self.Uxr = norm_weight(n_hidden, n_hidden, name='Uxr')
        self.bxr = zero_bias(n_hidden, 'bxr')
        # context vector
        self.Wc = norm_weight(n_hidden * 2, n_hidden, name='Wc')
        self.bc = zero_bias(n_hidden, 'bc')

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

        def _recurrence(x_t, xx_t, m, h_tm1, U,Ux):
            preact = x_t + T.dot(h_tm1, U)
            # reset fate
            r_t = T.nnet.sigmoid(split(preact, 0, self.n_hidden))
            # Update gate
            z_t = T.nnet.sigmoid(split(preact, 1, self.n_hidden))
            # Cell update
            c_t = T.tanh(T.dot(h_tm1, Ux) * r_t + xx_t)
            # Hidden state
            h_t = (T.ones_like(z_t) - z_t) * c_t + z_t * h_tm1
            # masking
            h_t = h_t * m[:, None]
            return h_t

        state_pre = T.zeros((self.x.shape[-1], self.n_hidden), dtype=theano.config.floatX)
        state_below = T.dot(self.x, self.W) + self.b
        state_belowx = T.dot(self.x, self.Wx) + self.bx
        h, _ = theano.scan(fn=_recurrence,
                           sequences=[state_below, state_belowx, self.mask],
                           non_sequences=[self.U,self.Ux],
                           outputs_info=state_pre,
                           truncate_gradient=-1)

        state_pre = T.zeros((self.x.shape[-1], self.n_hidden), dtype=theano.config.floatX)
        state_below = T.dot(self.x, self.Wr) + self.br
        state_belowx = T.dot(self.x, self.Wxr) + self.bxr
        hr, _ = theano.scan(fn=_recurrence,
                            sequences=[state_below, state_belowx, self.mask],
                            non_sequences=[self.Ur,self.Uxr],
                            go_backwards=True,
                            outputs_info=state_pre,
                            truncate_gradient=-1)

        # context will be the concatenation of forward and backward rnns
        ctx = T.concatenate([h, hr], axis=- 1)

        # mean of the context (across time) will be used to initialize decoder rnn
        ctx_mean = (ctx * self.mask[:, :, None]).sum(0) / self.mask.sum(0)[:, None]
        # or you can use the last state of forward + backward encoder rnns
        # ctx_last = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

        self.context = ctx_mean
        self.activation = T.tanh(T.dot(ctx_mean, self.Wc) + self.bc)

        '''
        if self.output_mode == 'sum':
            output = h + hr
        elif self.output_mode == 'concat':
            output = T.concatenate([h, hr], axs=-1)
        elif self.output_mode == 'mean':
            output = T.concatenate([h, hr], axs=-1)
        else:
            raise Exception("output mode is neither sum or concat")
        # Dropout
        if self.dropout > 0:
            drop_mask = self.rng.binomial(n=1, p=1 - self.dropout, size=output.shape, dtype=theano.config.floatX)
            self.activation = T.switch(self.is_train, output * drop_mask, output * (1 - self.dropout))
        else:
            self.activation = T.switch(self.is_train, output, output)
        '''
