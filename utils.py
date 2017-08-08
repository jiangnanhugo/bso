import gzip
import cPickle as pickle
import numpy as np
import h5py
import theano
import json
import os


def load_config(cfg_filename):
    '''Load a configuration file.'''
    with open(cfg_filename) as f:
        args = json.load(f)
    return args


def merge_dict(cfg_defaults, cfg_user):
    for k, v in cfg_defaults.items():
        if k not in cfg_user:
            cfg_user[k] = v
        elif isinstance(v, dict):
            merge_dict(v, cfg_user[k])


def load_config_with_defaults(cfg_filename, cfg_default_filename):
    """Load a configuration with defaults."""
    cfg_defaults = load_config(cfg_default_filename)
    cfg = load_config(cfg_filename)
    if cfg_filename != cfg_default_filename:
        merge_dict(cfg_defaults, cfg)
    return cfg


def save_model(f, model):
    output_folder = os.path.dirname(f)
    try:
        os.makedirs(output_folder)
    except Exception:
        pass
    ps = {}
    for p in model.params:
        ps[p.name] = p.get_value()
    pickle.dump(ps, open(f, 'wb'))


def load_model(f, model):
    ps = pickle.load(open(f, 'rb'))
    for p in model.params:
        p.set_value(ps[p.name])
    return model


def fopen(filepath, mode='r'):
    if filepath.endswith('.gz'):
        return gzip.open(filepath, mode)
    elif filepath.endswith('.hdf5'):
        return h5py.File(filepath, mode)
    elif filepath.endswith('.pkl'):
        return pickle.load(open(filepath, 'r'))
    else:
        vocab_dict = {}
        with open(filepath, mode)as f:
            for line in f:
                split_line = line.strip().split('\t')
                if len(split_line) != 2: break
                idx, word = split_line
                vocab_dict[word] = int(idx)
        return vocab_dict


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype(theano.config.floatX)


class TextIterator(object):
    def __init__(self, train_file, vocab_file, n_batch, maxlen=None):
        self.train_data = open(train_file, 'r')
        self.maxlen = maxlen
        self.n_batch = n_batch
        self.envocab = fopen(vocab_file[0])
        self.devocab = fopen(vocab_file[1])
        self.enidx2vocab = dict((v, k) for k, v in self.envocab.iteritems())
        self.deidx2vocab=dict((v,k) for k,v in self.devocab.iteritems())
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.train_data.seek(0)

    def goto_line(self, line_num):
        for _ in range(line_num):
            self.train_data.readline()

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        try:
            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    line = self.train_data.readline()
                    if line == '': break
                    splited_line = line.strip().split('#TAB#')
                    if len(splited_line) != 2: break
                    s, t = splited_line
                    s+=" <EOS>"
                    t="<START> "+t+" <EOS>"
                    s = s.split(' ')
                    t = t.split(' ')
                    if self.maxlen[0] > 0 and len(s) > self.maxlen[0]:
                        s = s[:self.maxlen[0]]
                    if self.maxlen[1] > 0 and len(t) > self.maxlen[1]:
                        t = t[:self.maxlen[1]]
                except IndexError:
                    break
                s = [self.envocab[w] if w in self.envocab else self.envocab['<UNK>'] for w in s]
                t = [self.devocab[w] if w in self.devocab else self.devocab['<UNK>'] for w in t]

                source.append(s)
                target.append(t)
                if len(source) >= self.n_batch and len(target) >= self.n_batch:
                    break
        except IOError:
            self.end_of_data = True

        # sort by target buffer
        tlen = np.array([len(t) for t in target])
        tidx = tlen.argsort()

        source = [source[i] for i in tidx]
        target = [target[i] for i in tidx]

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return prepare_data(source, target)


class BiTextIterator(object):
    def __init__(self, source, target, source_dict, target_dict, n_batch, maxlen=20):
        self.source = fopen(source)
        self.target = fopen(target)
        self.maxlen = maxlen
        self.n_batch = n_batch
        self.source_dict = fopen(source_dict)
        self.target_dict = fopen(target_dict)
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        try:
            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    s = self.source.readline()
                    t = self.target.readline()
                    if s == '' or t == '': break
                except IndexError:
                    break
                s = [self.source_dict[w] if w in self.source_dict else self.source_dict['unk'] for w in s]
                t = [self.target_dict[w] if w in self.target_dict else self.target_dict['unk'] for w in t]

                if self.maxlen > 0 and len(s) > self.maxlen and len(t) > self.maxlen:
                    continue
                source.append(s)
                target.append(t)
                if len(source) >= self.n_batch and len(target) >= self.n_batch:
                    break
        except IOError:
            self.end_of_data = True

        # sort by target buffer
        tlen = np.array([len(t) for t in target])
        tidx = tlen.argsort()

        source = [source[i] for i in tidx]
        target = [target[i] for i in tidx]

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target


def prepare_data(seqs_x, seqs_y):
    # x: a list of sentenxes
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) - 1 for s in seqs_y]

    n_batch = len(seqs_x)
    xmaxlen = np.max(lengths_x)
    ymaxlen = np.max(lengths_y)

    enc_input = np.zeros((xmaxlen, n_batch), dtype='int32')
    dec_input = np.zeros((ymaxlen, n_batch), dtype='int32')
    dec_output = np.zeros((ymaxlen, n_batch), dtype='int32')
    enc_mask = np.zeros((xmaxlen, n_batch), dtype=theano.config.floatX)
    dec_mask = np.zeros((ymaxlen, n_batch), dtype=theano.config.floatX)

    for idx, (sx, sy) in enumerate(zip(seqs_x, seqs_y)):
        enc_input[:lengths_x[idx], idx] = sx
        enc_mask[:lengths_x[idx], idx] = 1.
        dec_input[:lengths_y[idx], idx] = sy[:-1]
        dec_output[:lengths_y[idx], idx] = sy[1:]
        dec_mask[:lengths_y[idx], idx] = 1.
    input_list = [enc_input, enc_mask, dec_input, dec_output, dec_mask]
    return input_list
