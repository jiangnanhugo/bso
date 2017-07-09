# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import nltk
from nltk.corpus import brown
import numpy.random as random
from nltk import WordNetLemmatizer
from math import log
wnl=WordNetLemmatizer()
import cPickle as pickle
from tqdm import tqdm
epsilon=1e-8

def collect_data(filepath):
    texts=open(filepath,'r')

    linid=0
    fw=open(filepath+'pmi','w')
    for line in texts:
        linid+=1
        q,r=line.split('#TAB#')
        r=r.strip().split(' ')
        point=random.randint(low=1,high=len(r)-1,size=1)
        fw.write(q+'#TAB#'+r[:point]+'\n')

    fw.close()





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

collect_data('data/s2s_v3/s2s_v3_klrf/train.coarse.txt')

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