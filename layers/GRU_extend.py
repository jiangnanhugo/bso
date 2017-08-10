import theano
import theano.tensor as T
import numpy as np
from utils import *
from softmax import *


# attention decoder with schedule sampling
# https://github.com/nyu-dl/dl4mt-tutorial/tree/master/session3
class SSAttGRU(object):
    def __init__(self, srng, n_input, n_hidden, n_output,
                 x, mask, E, init_state=None, context=None, context_mask=None,
                 is_train=1, dropout=0.5,
                 one_step=False, dimctx=None, ss_prob=None):
        assert context, "Context must be provided"
        if one_step:
            assert init_state, 'previous state must be provided.'
        # projected context
        assert context.ndim == 3, \
            'Context must be 3-d: [batch_size, timesteps, hidden_size*2]'

        prefix = "AttGRU_"
        self.srng = srng

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.x = x
        self.mask = mask
        self.E = E
        self.ss_prob = ss_prob

        self.init_state = init_state
        self.context = context
        self.context_mask = context_mask
        self.is_train = is_train
        self.one_step = one_step
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
        self.U_nl = norm_weight(n_hidden, n_hidden * 2, name=prefix + 'U_nl')
        self.b_nl = zero_bias(n_hidden * 2, prefix + 'b_nl')
        # Cell update
        self.Ux_nl = norm_weight(n_hidden, n_hidden, name=prefix + 'Ux_nl')
        self.bx_nl = zero_bias(n_hidden, prefix + 'bx_nl')

        # context to lstm
        self.Wc = norm_weight(dimctx, n_hidden * 2, name=prefix + 'Wc')
        self.Wcx = norm_weight(dimctx, n_hidden, name=prefix + 'Wcx')
        # attention
        # context -> hidden
        self.W_comb_att = norm_weight(n_hidden, dimctx, name=prefix + 'W_comb_att')
        self.Wc_att = norm_weight(dimctx, name=prefix + 'Wc_att')
        self.U_att = norm_weight(dimctx, 1, name=prefix + 'U_att')
        self.b_att = zero_bias((dimctx,), name=prefix + 'b_att')  # hidden bias
        self.c_tt = zero_bias((1,), prefix + 'c_tt')

        self.sW = norm_weight(n_input, n_output, name=prefix + 'output_W')
        self.sU = norm_weight(dimctx, n_output, name=prefix + 'output_U')
        self.sb = zero_bias(n_output, name=prefix + 'output_b')

        # Params
        self.params = [self.W, self.U, self.b, self.Wx, self.Ux, self.bx,
                       self.U_nl, self.b_nl, self.Ux_nl, self.bx_nl,
                       self.Wc, self.Wcx, self.W_comb_att,
                       self.Wc_att, self.U_att, self.b_att, self.c_tt,
                       self.sW, self.sU, self.sb]
        self.build()

    def build(self):

        def split(x, n, dim):
            if x.ndim == 3:
                return x[:, :, n * dim: (n + 1) * dim]
            return x[:, n * dim:(n + 1) * dim]

        def _recurrence(x_t, m, ss_m,
                        h_, py_, y_t,
                        pctx_, cc_):
            # x_t: [1, 600], preact1: [100, 600]
            # scheduled sampling

            x_t = ss_m * x_t + (1 - ss_m) * y_t
            x_e = self.E[x_t, :]
            preact1 = T.nnet.sigmoid(T.dot(x_e, self.W) + T.dot(h_, self.U) + self.b)

            r1 = split(preact1, 0, self.n_hidden)
            u1 = split(preact1, 1, self.n_hidden)
            # r1: [100,300], u1:[100,300]
            h1 = T.tanh(T.dot(h_, self.Ux) * r1 + T.dot(x_e, self.Wx) + self.bx)
            h1 = u1 * h_ + (1. - u1) * h1
            h1 = m[:, None] * h1 + (1. - m)[:, None] * h_

            # attention
            # pstate_: 100 600
            pstate_ = T.dot(h1, self.W_comb_att)
            pctx__ = T.tanh(pstate_[None, :, :] + pctx_)

            # pctx__: 27 100 600
            # alpha: 27 100 1
            alpha = T.dot(pctx__, self.U_att) + self.c_tt
            alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
            # alpha 27 20
            alpha = T.exp(alpha)
            if self.context_mask:
                alpha = alpha * self.context_mask
            alpha = alpha / alpha.sum(0, keepdims=True)
            # alpha 27 20
            ctx_ = T.sum(cc_ * alpha[:, :, None], axis=0)  # batch_size, dimctx
            # ctx_ 20 600

            # another gru cell
            # ctx_:[batch_size,dimctx],
            # h1: [batch_size, hidden_size]
            preact2 = T.nnet.sigmoid(T.dot(h1, self.U_nl) + T.dot(ctx_, self.Wc) + self.b_nl)

            r2 = split(preact2, 0, self.n_hidden)
            u2 = split(preact2, 1, self.n_hidden)
            h2 = T.tanh((T.dot(h1, self.Ux_nl) + self.bx_nl) * r2 + T.dot(ctx_, self.Wcx))

            h2 = u2 * h1 + (1. - u2) * h2
            h2 = m[:, None] * h2 + (1. - m)[:, None] * h1

            logits = T.dot(h2, self.sW) + T.dot(ctx_, self.sU) + self.sb
            py_t = T.nnet.softmax(logits)
            y_t = T.argmax(py_t, axis=-1)
            return h2, py_t, y_t

        # self.init_state: [100,300]
        if self.init_state == None:
            self.init_state = T.zeros((self.x.shape[1], self.n_hidden), dtype=theano.config.floatX)
        # mask
        # self.mask: [1,1]
        if self.mask == None:
            self.mask = T.alloc(1., self.x.shape[0], 1)

        # pctx_:[seqlen,batch_size,dimctx]
        # self.context:[27,20,600]
        # pctx: [27,20,600]
        pctx_ = T.dot(self.context, self.Wc_att) + self.b_att
        ss_mask = self.srng.binomial(self.x.shape, p=self.ss_prob)

        seqs = [self.x, self.mask, ss_mask]
        if self.one_step == True:
            seqs += [self.init_state, None, pctx_, self.context]
            _, activation, predict = _recurrence(*seqs)
        else:
            [_, activation, predict], _ = theano.scan(fn=_recurrence,
                                                      sequences=seqs,
                                                      non_sequences=[pctx_, self.context],
                                                      outputs_info=[self.init_state,
                                                                    T.alloc(0., self.context.shape[1], self.n_output),
                                                                    T.zeros((self.x.shape[1],),
                                                                            dtype='int64')])
            # T.alloc(0,self.context.shape[1])])

        self.activation = activation
        self.predict = predict
