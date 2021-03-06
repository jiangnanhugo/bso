import time
from seq2seq import *
from utils import *

import logging
from logging.config import fileConfig
import numpy
numpy.set_printoptions(threshold=numpy.nan)

fileConfig('logging_config.ini')
logger = logging.getLogger()

from argparse import ArgumentParser

argument = ArgumentParser(usage='it is usage tip', description='no')
argument.add_argument('--cfgfile', default='./configurations/model_topk.json', type=str, help='model config')
argument.add_argument('--basic_cfgfile', default='./configurations/basic.json', type=str, help='basic model config')

arguments = argument.parse_args()
args = load_config_with_defaults(arguments.cfgfile, arguments.basic_cfgfile)
print args

train_datafile = args['train_file']
vocab_file = args['vocab_file']
valid_datafile = args['valid_file']
test_datafile = args['test_file']
checkpoint = args['checkpoint']
goto_line = args['goto_line']
n_batch = args['batch_size']
rnn_cells = args['rnn_cells']  # list
optimizer = args['optimizer']
clip_freq = args['clip_freq']
maxlen = args['maxlen']
disp_freq = 100
NEPOCH = args['epochs']
# valid_freq = args['valid_freq']
# test_freq = args['test_freq']
# save_freq = args['save_freq']
n_input = args['n_input']  # embedding of input word
n_hidden = args['n_hidden']  # hidden state layer size
dropout = args['dropout']
lr = args['learning_rate']
mode = args['mode']
k = args['beam']


def train():
    logger.info('loading dataset...')
    train_data = TextIterator(train_datafile, vocab_file, n_batch, maxlen)
    logger.info('building model...')
    envocab_size = len(train_data.envocab)
    devocab_size = len(train_data.devocab)
    model = Seq2Seq(envocab_size, devocab_size, n_hidden, rnn_cells, optimizer, dropout)
    if os.path.isfile(checkpoint):
        print 'loading checkpoint parameters....', checkpoint
        model = load_model(checkpoint, model)
    if goto_line > 0:
        train_data.goto_line(goto_line)
        logger.info('goto line: %d' % goto_line)
    logger.info('training start...')
    start = time.time()
    idx = goto_line
    epsilon=0.25
    for epoch in xrange(NEPOCH):
        batch_cost = 0
        batch_acc = 0
        for input_list in train_data:
            idx += 1
            input_list.append(lr)
            input_list.append(epsilon)
            cost, acc = model.train(*input_list)
            batch_cost += cost
            batch_acc += acc
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN Or Inf detected!'
                return -1
            if idx % disp_freq == 0:
                logger.info(
                    'epoch: %d idx: %d cost: %f acc: %f' % (epoch, idx, batch_cost / disp_freq, batch_acc / disp_freq))
                batch_cost = 0
                batch_acc = 0
        epsilon*=0.9
        logger.info("Clip the epsilon of schedule sampling to: %f" %epsilon)
        logger.info('dumping with epoch %d' % epoch)
        prefix = './model/epoch_%d_time_%.2f.pkl' % (epoch, (time.time() - start))
        save_model(prefix, model)

    logger.info("Finished. Time = %s" % str(time.time() - start))


def test():
    test_data = TextIterator(test_datafile, vocab_file, n_batch=1, maxlen=maxlen)
    envocab_size = len(test_data.envocab)
    devocab_size = len(test_data.devocab)
    deidx2vocab=test_data.deidx2vocab
    enidx2vocab=test_data.enidx2vocab
    vocab2idx=test_data.devocab
    model = Seq2Seq(envocab_size, devocab_size, n_hidden, rnn_cells, optimizer, dropout,one_step=1)
    if os.path.isfile(checkpoint):
        print 'loading pretrained model:', checkpoint
        model = load_model(checkpoint, model)
    else:
        print checkpoint, 'not found'
        raise Exception('pretrained model file must be provided!')

    for input_list in test_data:
        sample = []
        sample_score = []
        length = maxlen[1]
        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k, dtype=theano.config.floatX)

        GO_ID=vocab2idx['<START>']
        next_w =  np.asarray([GO_ID,], dtype='int32')
        next_state, ctx0 = model.encoder_state(input_list[0],input_list[1])

        for w in input_list[0].flatten():
            print enidx2vocab[w],
        print
        for i in range(length):
            ctx = np.tile(ctx0, [live_k, 1])
            next_p,next_state=model.predict(next_w,ctx,next_state)

            cand_scores = hyp_scores[:, None] - np.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k - dead_k)]

            trans_indices = ranks_flat / devocab_size
            word_indices = ranks_flat % devocab_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = np.zeros(k - dead_k, dtype=theano.config.floatX)
            new_hyp_states = []
            for idx, (ti, wi) in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_hyp_scores[idx] = costs[idx]
                new_hyp_states.append(next_state[ti])

            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            for idx in range(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = np.array(hyp_scores)
            live_k = new_live_k
            if new_live_k < 1 or dead_k >= k:
                break

            next_w = np.asarray([w[-1] for w in hyp_samples],dtype='int32')
            next_state = np.array(hyp_states)


        if live_k > 0:
            for idx in range(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
        print 'Beam search results:'
        for sent in sample:
            for w in sent:
                print deidx2vocab[w],
            print


def topk():
    logger.info('loading dataset...')
    test_data = TextIterator(test_datafile, vocab_file, n_batch, maxlen)
    logger.info('building model...')
    envocab_size = len(test_data.envocab)
    devocab_size = len(test_data.devocab)
    deidx2vocab = test_data.deidx2vocab
    model = Seq2Seq(envocab_size, devocab_size, n_hidden, rnn_cells, optimizer, dropout,one_step=2)
    if os.path.isfile(checkpoint):
        print 'loading checkpoint parameters....', checkpoint
        model = load_model(checkpoint, model)
    if goto_line > 0:
        test_data.goto_line(goto_line)
        logger.info('goto line: %d' % goto_line)
    logger.info('training start...')

    for input_list in test_data:
        topked,y,y_mask = model.topk(*input_list)
        y=y.flatten()
        y_mask=y_mask.flatten()
        rol,col=topked.shape
        for i in range(rol):
            print deidx2vocab[y[i]],topked[i][y[i]]
        print '='*40




if __name__ == '__main__':
    if mode == 'train':
        train()
    elif mode == 'testing':
        test()
    elif mode=='topk':
        topk()
