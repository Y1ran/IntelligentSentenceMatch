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
    #加载模型到内存中
    global model
    model = KeyedVectors.load_word2vec_format('pretrainModel\\merge_sgns_bigram_char300.txt', binary=False) 
    model.wv.save_word2vec_format("pretrainModel\\pretrain_word2vec_modelV1.vector")

if __name__ == "__main__":
    load_model()
    print("模型已经转换结束")