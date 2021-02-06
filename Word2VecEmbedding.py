# coding=GBK

from pyhanlp import *
import traceback
import warnings
import numpy as np
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
from gensim.models import KeyedVectors
from scipy import spatial
from flask import Flask
import sys

app = Flask(__name__)

model=None

def load_model():
    #加载模型到内存中
    global model
    model = KeyedVectors.load_word2vec_format('pretrainModel\\pretrain_word2vec_modelV1.vector', binary=False) 

class Word2VecTester():

    def __init__(self):
        model = Word2Vec(sentences, sg=1, size=100, window=5, min_count=5, negative=3, sample=0.001, hs=1, workers=4)

    def filtered_punctuations(self, token_list):
        try:
            punctuations = ['']
            token_list_without_punctuations = [word for word in token_list
                                                             if word not in punctuations]
            #print "[INFO]: filtered_punctuations is finished!"
            return token_list_without_punctuations

        except Exception as e:
            print (traceback.print_exc())


    def list_crea(self, everyone):
        list_word = []
        for k in everyone:
            fenci= filtered_punctuations(k)
            list_word.append(fenci)

        return list_word

class keywordMatchScore():

    def __init__(self):
        self.listpath = None
        self.sentence = None

    # 创建停用词列表
    def stopwordslist(self,listpath):
	    stopwords = [line.strip() for line in open(listpath, encoding='GBK').readlines()]
	    return stopwords


    # 对句子进行分词
    def seg_sentence(self,sentence):
        sentence_seged = HanLP.segment(sentence.strip())
        stopwords = self.stopwordslist('stopwords.txt')  # 这里加载停用词的路径
        outstr = ''
        print("hanlp outputs: ", sentence_seged)
        for word in sentence_seged:
            if word.word not in stopwords:
                if word != '\t':
                    outstr += word.word
                    outstr += " "
        return outstr

    # 计算关键词匹配得分
    def score_compute(self, inputs1, inputs2):
        score = len(inputs1.intersection(inputs2)) / max(len(inputs1), len(inputs2)) 

        return score



def simlarityCalu(vector1,vector2):
    #计算余弦相似度
    vector1Mod = np.sqrt(vector1.dot(vector1))
    vector2Mod = np.sqrt(vector2.dot(vector2))

    if vector2Mod!=0 and vector1Mod!=0:
        simlarity=(vector1.dot(vector2))/(vector1Mod*vector2Mod)
    else:
        simlarity=0
    return simlarity

if __name__ == '__main__':

    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    #app.run()
    scoreMatcher = keywordMatchScore()
    inputs = open('knowledge\\test_input.txt', 'r') #加载要处理的文件的路径
    outputs = open('test_output.txt', 'w') #加载处理后的文件路径
    line_outputs = []
    embeddings_index = {}

    print('model sim: ', model.similarity('电脑', '计算机'))
    print('model sim: ', model.similarity('陕西', '陕西省'))
    print('model sim: ', model.similarity('计算器', '计算机'))

    for line in inputs:
        print("关键词：", HanLP.extractKeyword(line, 4))
        # 自动摘要
        print("摘要句：", HanLP.extractSummary(line, 2))
        line_seg = scoreMatcher.seg_sentence(line).strip()  # 这里的返回值是字符串
        line_outputs.append(line_seg.split(' '))
        print("line_outputs: ", line_outputs)
    
    cos_input = {}
    cos_output = []
    for sentence in line_outputs:
        avg_vec = []
        for token in sentence:
            #print("word ", token, "with embedding vector: ", model[token])#, embeddings_index[word])
            avg_vec.append(model[token])
        avg_embedding_test = np.mean(avg_vec,axis=0)
        print("average embedding: ", avg_embedding_test)
        cos_input[''.join(sentence)] = avg_embedding_test
        cos_output.append(avg_embedding_test)

    #index2word_set = set(model.wv.index2word)
    embeddings = 'knowledge\\test_embedding.txt' #加载知识库向量存储的文件路径
    with open(embeddings, 'w') as file_object:
        for k,v in cos_input.items():
            file_object.write(k + '\t'+ ','.join([str(x) for x in v])+'\t'
            + k对应的答案)

    print("完成预训练向量的加载")