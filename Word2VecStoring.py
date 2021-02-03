# coding=GBK

import traceback
import warnings
import numpy as np
from gensim.models import KeyedVectors
from flask import Flask
import sys

app = Flask(__name__)

model=None

def load_model():
    #����ģ�͵��ڴ���
    global model
    model = KeyedVectors.load_word2vec_format('merge_sgns_bigram_char300.txt', binary=False) 
    model.wv.save("pretrianModel\\pretrain_word2vec_modelV1.wv")

if __name__ == "__main__":
    load_model()
    print("ģ���Ѿ�ת������")