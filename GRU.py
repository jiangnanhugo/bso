import numpy as np
import theano
import theano.tensor as T


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

        self.inti_state=init_state
        self.is_train = is_train
        self.p = p

        # Update gate
        init_W = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_input, n_hidden * 2)),
                            dtype=theano.config.floatX)
        init_U = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_hidden, n_hidden*2)),
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
        if self.inti_state==None:
            self.init_state = T.zeros((self.x.shape[-1], self.n_hidden), dtype=theano.config.floatX)
        state_below = T.dot(self.E[self.x,:], self.W) + self.b
        state_belowx = T.dot(self.E[self.x,:], self.Wx) + self.bx

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
                 x, E, mask, init_state=None,context=None,context_mask=None,
                 is_train=1, p=0.5):
        assert context, "Context must be provided"

        # https://github.com/nyu-dl/dl4mt-tutorial/tree/master/session3
        self.rng = rng

        self.n_input = n_input
        self.n_hidden = n_hidden

        self.x = x
        self.E = E
        self.mask = mask


        self.init_state=init_state
        self.context=context
        self.context_mask = context_mask
        self.is_train = is_train
        self.p = p

        # Update gate
        init_W = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_input, n_hidden * 2)),
                            dtype=theano.config.floatX)
        init_U = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_hidden, n_hidden*2)),
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
        if self.init_state==None:
            self.init_state = T.zeros((self.x.shape[-1], self.n_hidden), dtype=theano.config.floatX)

        # projected context
        assert  self.context.ndim==3,\
            'Context must be 3-d: # annotation x #sample x dim'


        # projected x
        state_below = T.dot(self.E[self.x,:], self.W) + self.b
        state_belowx = T.dot(self.E[self.x,:], self.Wx) + self.bx

        def split(x, n, dim):
            if x.ndim == 3:
                return x[:, :, n * dim: (n + 1) * dim]
            return x[:, n * dim:(n + 1) * dim]

        def align(h_t,h_s,sim='dot'):
            if sim=='dot':
                score= T.dot(h_t,h_s)
            elif sim=='general':
                score=0
            elif sim=='concat':
                score=0
            else:
                raise Exception("similarity function must be dot or general or concat.")

            shape=score.shape
            score=score.reshape((-1,shape[-1]))
            alpha=T.nnet.softmax(score)
            return alpha.reshape(shape)



        def _recurrence(x_t, xx_t, m, h_tm1,
                        ctx_t, alpha_t,pctx_t,cc_t,
                        U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
                        U_nl, Ux_nl, b_nl, bx_nl):
            preact = T.nnet.sigmoid(x_t + T.dot(h_tm1, U))

            # reset fate
            r_t = split(preact, 0, self.n_hidden)
            # Update gate
            z_t = split(preact, 1, self.n_hidden)
            # Cell update
            c_t = T.tanh(T.dot(h_tm1, Ux) * r_t + xx_t)
            # Hidden state
            h_t = z_t * h_tm1 + (1. - z_t) * c_t
            # masking
            h_t = h_t * m[:, None] + (1. - m)[:, None] * h_tm1

            # attention
            alpha=T.dot(T.tanh(T.dot(h_t,W_comb_att) + pctx_t ), U_att) + c_tt
            alpha=alpha.reshape([alpha.shape[0],-1])
            alpha=T.exp(alpha)
            if self.context_mask:
                alpha=T.exp(alpha)*self.context_mask
            alpha=alpha/alpha.sum(0,keepdims=True)
            ctx_=(cc_t * alpha).sum(0)


            # another gru cell
            preact2 = T.dot(h_t,l) + T.dot(ctx_t, Wc)+ b_nl

            # reset fate # Update gate
            r_t2 = T.nnet.sigmoid(split(preact2, 0, self.n_hidden))
            z_t2 = T.nnet.sigmoid(split(preact2, 1, self.n_hidden))
            # Cell update
            c_t2 = T.tanh((T.dot(h, Ux_nl) +bx_nl)* r_t2 + T.dot(ctx_,Wcx))
            # Hidden state
            h_t2 = (1. - z_t2) * c_t2 + z_t2 * h_t
            # masking
            h_t2 = h_t2 * m[:, None] + (1. - m)[:, None] * h_t

            return h_t2, ctx_,alpha.T

        seqs=[state_below, state_belowx, self.mask]
        h, _ = theano.scan(fn=_recurrence,
                           sequences=seqs,
                           non_sequences=[context,self.U,self.Ux],
                           outputs_info=self.init_state,
                           truncate_gradient=-1)

        # Dropout
        if self.p > 0:
            drop_mask = self.rng.binomial(n=1, p=1 - self.p, size=h.shape, dtype=theano.config.floatX)
            self.activation = T.switch(self.is_train, h * drop_mask, h * (1 - self.p))
        else:
            self.activation = T.switch(self.is_train, h, h)



class BiGRU(object):
    def __init__(self, rng,n_input, n_hidden,
                 x, mask, output_mode='sum',
                 is_train=1, dropout=0.5):
        # https://github.com/nyu-dl/dl4mt-tutorial/tree/master/session3
        self.rng = rng

        self.n_input = n_input
        self.n_hidden = n_hidden

        self.x = x
        self.mask = mask

        self.output_mode=output_mode
        self.is_train = is_train
        self.dropout = dropout

        # Update gate
        init_W = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_input, n_hidden * 2)),
                            dtype=theano.config.floatX)
        init_U = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_hidden, n_hidden*2)),
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

        # backword
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

        self.Wr = theano.shared(value=init_W, name='Wr')
        self.Ur = theano.shared(value=init_U, name='Ur')
        self.br = theano.shared(value=init_b, name='br')

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

        self.Wxr = theano.shared(value=init_Wx, name='Wxr')
        self.Uxr = theano.shared(value=init_Ux, name='Uxr')
        self.bxr = theano.shared(value=init_bx, name='bxr')

        # Params
        self.params = [self.W, self.U, self.b, self.Wx, self.Ux, self.bx,
                       self.Wr, self.Ur, self.br, self.Wxr, self.Uxr, self.bxr]

        self.build_graph()

    def build_graph(self):


        def split(x, n, dim):
            if x.ndim == 3:
                return x[:, :, n * dim: (n + 1) * dim]
            return x[:, n * dim:(n + 1) * dim]

        def _recurrence(x_t, xx_t, m, h_tm1,Ux):
            preact = x_t + T.dot(h_tm1, self.U)

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
                           non_sequences=[self.Ux],
                           outputs_info=state_pre,
                           truncate_gradient=-1)

        state_pre = T.zeros((self.x.shape[-1], self.n_hidden), dtype=theano.config.floatX)
        state_below = T.dot(self.x, self.Wr) + self.br
        state_belowx = T.dot(self.x, self.Wxr) + self.bxr
        hr, _ = theano.scan(fn=_recurrence,
                            sequences=[state_below, state_belowx, self.mask],
                            non_sequences=[self.Uxr],
                            go_backwards=True,
                            outputs_info=state_pre,
                            truncate_gradient=-1)

        if self.output_mode=='sum':
            output=h+hr
        elif self.output_mode=='concat':
            output=T.concatenate([h,hr],axs=-1)
        elif self.output_mode=='mean':
            output = T.concatenate([h, hr], axs=-1)
        else:
            raise Exception("output mode is neither sum or concat")


        self.context=T.concatenate([h,hr],axs=-1)
        # Dropout
        if self.dropout > 0:
            drop_mask = self.rng.binomial(n=1, p=1 - self.dropout, size=output.shape, dtype=theano.config.floatX)
            self.activation = T.switch(self.is_train, output * drop_mask, output * (1 - self.dropout))
        else:
            self.activation = T.switch(self.is_train, output, output)