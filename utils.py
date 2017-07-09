import gzip
import cPickle as pickle
import numpy as np
import h5py
import theano
def fopen(filepath,mode='r'):
    if filepath.endswith('.gz'):
        return gzip.open(filepath,mode)
    elif filepath.endswith('.hdf5'):
        return h5py.File(filepath,mode)
    elif filepath.endswith('.hdf5'):
        return pickle.load(open(filepath,'r'))
    return open(filepath,mode)



def init_norm(n_in,n_out):
    init_W=np.asarray(np.random.randn(n_in,n_out)*0.1,dtype=theano.config.floatX)
    return init_W

class TextIterator(object):
    def __init__(self,source,target,source_dict,target_dict,n_batch,maxlen=None):
        self.source=fopen(source)
        self.target=fopen(target)
        self.maxlen=maxlen
        self.source_dict=fopen(source_dict)
        self.target_dict=fopen(target_dict)
        self.end_of_data=False


    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data=False
            self.reset()
            raise StopIteration

        source=[]
        target=[]

        try:
            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    s = self.source.readline()
                    if s == '': break
                    t = self.target.readline()
                    if t == "": break
                except IndexError:
                    break
                s=[self.source_dict[w] if w in self.source_dict else self.source_dict['unk'] for w in s]

                t=[self.target_dict[w] if w in self.target_dict else self.target['unk'] for w in t]

                if len(s)>self.maxlen and len(t)>self.maxlen:
                    continue
                source.append(s)
                target.append(t)
                if len(source)>=self.n_batch and len(target)>=self.n_batch:
                    break
        except IOError:
            self.end_of_data=True

        # sort by target buffer
        tlen = np.array([len(t) for t in target])
        tidx = tlen.argsort()

        source = [source[i] for i in tidx]
        target = [target[i] for i in tidx]


        if len(source)<=0 or len(target)<=0:
            self.end_of_data=False
            self.reset()
            raise StopIteration

        return source, target