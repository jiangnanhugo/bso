# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import nltk
from nltk.corpus import brown
import numpy.random as random
from nltk import WordNetLemmatizer
from math import log
import numpy as np
wnl=WordNetLemmatizer()
import cPickle as pickle
from collections import defaultdict
epsilon=1e-8

def collect_data(filepath,envocab,devocab):
    texts=open(filepath,'r')
    enfile=open(envocab,'r').read().split('\n')
    endict=set()
    for line in enfile:
        splited=line.split('\t')
        if len(splited)!=2: break
        endict.add(splited[1])

    defile = open(devocab, 'r').read().split('\n')
    dedict = set()
    for line in defile:
        splited = line.split('\t')
        if len(splited) != 2: break
        dedict.add(splited[1])


    linid=0
    qlen=[]
    rlen=[]
    qset=0
    rset=0
    for line in texts:
        linid+=1
        q,r=line.split('#TAB#')
        q=q.strip().split(' ')
        for w in q:
            if w not in endict:
                qset+=1

        r=r.strip().split(' ')
        for w in r:
            if w not in dedict:
                rset+=1
        qlen.append(len(q))
        rlen.append(len(r))
    print 'line number',linid
    print 'query words',np.sum(qlen)
    print 'response words', np.sum(rlen)
    print len(endict),qset,len(dedict),rset










def pmi_keywords(filepath):
    with open('_Fdist.pkl', 'r')as f:
        _Fdist=pickle.load(f)
    with open('word.pkl', 'r')as f:
        word=pickle.load(f)

    with open('sents.pkl', 'r')as f:
        _Sents=pickle.load( f)


    def p(x):
        return (_Fdist[x]+epsilon)/float(len(_Fdist))

    def pxy(x,y):
        return (len(filter(lambda s: x in s and y in s, _Sents))+1)/float(len(_Sents))

    def pmi(x,y):
        return log(pxy(x,y)/(p(x)*p(y)),2)

    pair=dict()

    index=0
    for w1 in word:
        index+=1
        if index%1000==0:
            print index,
        for w2 in word:
            if w1!=w2 and (w1,w2) not in pair:
                score=pmi(w1, w2)
                pair[(w1,w2)]=score


    with open('pairs.pkl','w')as f:
        pickle.dump(pair,f)

collect_data('data/s2s_v3/s2s_v3_klrf/train.coarse.txt',
             'data/vocab_v3/vocab_v3_klrf/s2s.model.vocab.enc.coarse.txt',
             'data/vocab_v3/vocab_v3_klrf/s2s.model.vocab.dec.coarse.txt')

#pmi_keywords('data/s2s_v3/s2s_v3_klrf/train.coarse.txt')
'''
with open('words.pkl','r')as f:
    words=pickle.load(f)

word=set()
for w in words:
    word.add(w)

_Fdist=nltk.FreqDist(words)


with open('word.pkl', 'w')as f:
    pickle.dump(word, f)

with open('_Fdist.pkl', 'w')as f:
    pickle.dump(_Fdist, f)
'''