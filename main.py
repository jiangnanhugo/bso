import time
from seq2seq import *
from utils import *

import logging
from logging.config import fileConfig

fileConfig('logging_config.ini')
logger = logging.getLogger()

from argparse import ArgumentParser

argument = ArgumentParser(usage='it is usage tip', description='no')
argument.add_argument('--cfgfile', default='./configurations/extended.json', type=str, help='model config')
argument.add_argument('--basic_cfgfile', default='./configurations/basic.json', type=str, help='model config')

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
maxlen = args['maxlen']
disp_freq = 100
NEPOCH = args['epochs']
#valid_freq = args['valid_freq']
#test_freq = args['test_freq']
#save_freq = args['save_freq']
n_input = args['n_input']  # embedding of input word
n_hidden = args['n_hidden']  # hidden state layer size
dropout = args['dropout']
lr = args['learning_rate']
mode=args['mode']


def evaluate(test_data, model):
    sumed_cost = 0
    sumed_wer = []
    n_words = []
    idx = 0
    for x, x_mask, y, y_mask in test_data:
        # nll,pred_y=model.test(x,x_mask,y,y_mask)
        # sumed_wer.append(calculate_wer(y,y_mask,np.reshape(pred_y, y.shape)))
        sumed_wer.append(1.)
        sumed_cost += 1.0
        idx += 1  # np.sum(y_mask)
        # n_words.append(np.sum(y_mask))
        n_words.append(1.)
    return sumed_cost / (1.0 * idx), np.sum(sumed_wer) / np.sum(n_words)


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
    for epoch in xrange(NEPOCH):
        batch_cost = 0
        batch_acc = 0
        for input_list in train_data:
            idx += 1
            input_list.append(lr)
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
        logger.info('dumping with epoch %d' % epoch)
        prefix = './model/epoch_%d_time_%.2f.pkl' % (epoch, (time.time() - start))
        save_model(prefix, model)

    logger.info("Finished. Time = %s" % str(time.time() - start))


def test():
    test_data = TextIterator(test_datafile, vocab_file, n_batch, maxlen)
    valid_data = TextIterator(valid_datafile, vocab_file, n_batch, maxlen)
    envocab_size = len(valid_data.envocab)
    devocab_size = len(valid_data.devocab)
    model = Seq2Seq(envocab_size, devocab_size, n_hidden, rnn_cells, optimizer, dropout)
    if os.path.isfile(args.model_dir):
        print 'loading pretrained model:', args.model_dir
        model = load_model(args.model_dir, model)
    else:
        print args.model_dir, 'not found'
    mean_cost = evaluate(valid_data, model)
    print 'valid cost:', mean_cost, 'perplexity:', np.exp(mean_cost)  # ,"word_error_rate:",mean_wer
    mean_cost = evaluate(test_data, model)
    print 'test cost:', mean_cost, 'perplexity:', np.exp(mean_cost)


if __name__ == '__main__':
    if mode == 'train':
        train()
    elif mode == 'testing':
        test()
