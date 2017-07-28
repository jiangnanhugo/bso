import numpy as np
import theano
import theano.tensor as T

import logging

logging.basicConfig(level=logging.DEBUG)
logger=logging.getLogger(__name__)

# generate sample with beam search.
# note that, this function iteratively calls f_init and f_next functions.

def gen_sample(tparams,f_init,f_next,options,trng=None,k=1,maxlen=30):
    sample=[]
    sample_score=[]
    live_k=1
    dead_k=0

    for i in range(maxlen):
        ctx=np.tile(ctx0,[live_k,1])
        inps=[next_w,ctx,next_state]
        ret=f_next(*inps)
        next_p,next_w,next_state=ret[0],ret[1],ret[2]

        cand_scores=hyp_scores[:,None]-np.log(next_p)
        cand_flat=cand_scores.flatten()
        ranks_flat.argsort()[:(k-dead_k)]

        voc_size=next_p.shape[1]
        trans_indices=ranks_flat/voc_size
        word_indices=ranks_flat%voc_size
        costs-cand_flat[ranks_flat]

