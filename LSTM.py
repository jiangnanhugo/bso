import numpy as np
import theano
import theano.tensor as T


class LSTM(object):
    """
    LSTM with faster implementatin.
    """

    def __init__(self,rng,n_input,n_hidden,
                 x,E,mask,init_state=None,is_train=1,p=0.5):
        self.rng=rng

        self.n_input=n_input
        self.n_hidden=n_hidden

        self.x=x
        self.E=E
        self.mask=mask
        self.is_train=is_train
        self.p=p
        self.f=T.nnet.sigmoid
        self.init_state=init_state

        init_W=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_input,n_hidden*4)),
                           dtype=theano.config.floatX)
        init_U = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_hidden, n_hidden * 4)),
                            dtype=theano.config.floatX)

        init_b=np.zeros((n_hidden*4),dtype=theano.config.floatX)

        self.W=theano.shared(value=init_W,name="W")
        self.U = theano.shared(value=init_U, name="U")
        self.b = theano.shared(value=init_b, name="b")

        self.params=[self.W,self.U,self.b]

        self.build()



    def build(self):
        def split(x,n,dim):
            return x[:,n*dim:(n+1)*dim]

        def _recurrence(x_t,m,h_tm1,c_tm1):
            p=x_t+T.dot(h_tm1,self.U)
            # Input Gate
            i_t = self.f(split(p,0,self.n_hidden))
            # Forget Gate
            f_t = self.f(split(p,1,self.n_hidden))
            # Output Gate
            o_t = self.f(split(p,2,self.n_hidden))
            # Cell update
            c_tilde_t=T.tanh(split(p,3,self.n_hidden))
            c_t=f_t * c_tm1 + i_t * c_tilde_t
            # Hidden State
            h_t = o_t * T.tanh(c_t)

            c_t=c_t * m[:,None]
            h_t=h_t * m[:,None]

            return [h_t,c_t]

        pre=T.dot(self.E[self.x,:],self.W)+self.b
        if self.init_state==None:
            self.init_state=[dict(initial=T.zeros((self.x.shape[-1], self.n_hidden))),dict(initial=T.zeros((self.x.shape[-1], self.n_hidden)))]

        [h,c],_=theano.scan(fn=_recurrence,
                            sequences=[pre,self.mask],
                            outputs_info=self.init_state)
        if self.p>0:
            drop_mask=self.rng.binomial(n=1,p=1-self.p,size=h.shape,dtype=theano.config.floatX)
            self.activation=T.switch(self.is_train,h*drop_mask,h*(1-self.p))
        else:
            self.activation=T.switch(self.is_train,h,h)


class BiLSTM(object):
    """
    BiLSTM with for encoder of seq2seq model.
    """

    def __init__(self,rng,n_input,n_hidden,
                 x,E,mask,output_mode='sum',is_train=1,p=0.5):
        self.rng=rng

        self.n_input=n_input
        self.n_hidden=n_hidden

        self.x=x
        self.E=E
        self.mask=mask

        self.output_mode=output_mode
        self.is_train=is_train
        self.p=p
        self.f=T.nnet.sigmoid

        # f: forward,
        init_Wf=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                            high=np.sqrt(1./n_input),
                                            size=(n_input,n_hidden*4)),
                           dtype=theano.config.floatX)
        init_Uf = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                              high=np.sqrt(1. / n_input),
                                              size=(n_hidden, n_hidden * 4)),
                            dtype=theano.config.floatX)

        init_bf=np.zeros((n_hidden*4),dtype=theano.config.floatX)

        self.Wf=theano.shared(value=init_Wf,name="Wf")
        self.Uf = theano.shared(value=init_Uf, name="Uf")
        self.bf = theano.shared(value=init_bf, name="bf")

        #r: reverse
        init_Wr = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                               high=np.sqrt(1. / n_input),
                                               size=(n_input, n_hidden * 4)),
                             dtype=theano.config.floatX)
        init_Ur = np.asarray(np.random.uniform(low=-np.sqrt(1. / n_input),
                                               high=np.sqrt(1. / n_input),
                                               size=(n_hidden, n_hidden * 4)),
                             dtype=theano.config.floatX)

        init_br = np.zeros((n_hidden * 4), dtype=theano.config.floatX)

        self.Wr = theano.shared(value=init_Wr, name="Wr")
        self.Ur = theano.shared(value=init_Ur, name="Ur")
        self.br = theano.shared(value=init_br, name="br")

        self.params=[self.Wf,self.Uf,self.bf,self.Wr,self.Ur,self.br]

        self.build()



    def build(self):
        def split(x,n,dim):
            return x[:,n*dim:(n+1)*dim]

        def _recurrence(x_t,m,h_tm1,c_tm1,U):
            p=x_t+T.dot(h_tm1,U)
            # Input Gate
            i_t = self.f(split(p,0,self.n_hidden))
            # Forget Gate
            f_t = self.f(split(p,1,self.n_hidden))
            # Output Gate
            o_t = self.f(split(p,2,self.n_hidden))
            # Cell update
            c_tilde_t=T.tanh(split(p,3,self.n_hidden))
            c_t=f_t * c_tm1 + i_t * c_tilde_t
            # Hidden State
            h_t = o_t * T.tanh(c_t)

            c_t=c_t * m[:,None]
            h_t=h_t * m[:,None]

            return [h_t,c_t]

        pre=T.dot(self.E[self.x,:],self.Wf)+self.bf

        # forward process
        [hf,cf],_=theano.scan(fn=_recurrence,
                            sequences=[pre,self.mask],
                            non_sequences=[self.Uf],
                            outputs_info=[dict(initial=T.zeros((self.x.shape[-1],self.n_hidden))),
                                          dict(initial=T.zeros((self.x.shape[-1],self.n_hidden)))])

        # backward process
        pre = T.dot(self.E[self.x, :], self.Wr) + self.br
        [hr, cr], _ = theano.scan(fn=_recurrence,
                                sequences=[pre, self.mask],
                                non_sequences=[self.Ur],
                                go_backwards=True,
                                outputs_info=[dict(initial=T.zeros((self.x.shape[-1], self.n_hidden))),
                                              dict(initial=T.zeros((self.x.shape[-1], self.n_hidden)))])

        if self.output_mode=='sum':
            output=hf+hr
        elif self.output_mode=='concat':
            output=T.concatenate([hf,hr],axis=-1)
        elif self.output_mode=="mean":
            raise NotImplementedError('mean function not implemented yet!')
        else:
            raise  Exception("output mode is neither sum or concat")


        if self.p>0:
            drop_mask=self.rng.binomial(n=1,p=1-self.p,size=output.shape,dtype=theano.config.floatX)
            self.activation=T.switch(self.is_train,output*drop_mask,output*(1-self.p))
        else:
            self.activation=T.switch(self.is_train,output,output)


