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
            if word not in stopwords:
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
    scoreMatcher = keywordMatchScore()
    #app.run()
    knowledgeBase = {}
    embeddings = 'knowledge\\test_embedding.txt' #加载知识库向量存储的文件路径
    with open(embeddings, 'r') as file_object:
        for line in file_object:
            knowledgeBase[line.split(':')[0]] = line.split(':')[1].split(',')
            print('model input embedding: ', line)
    

    #index2word_set = set(model.wv.index2word)

    #s1_afv = avg_feature_vector(line_outputs[0], model=model, num_features=300, index2word_set=index2word_set)
    #s2_afv = avg_feature_vector(line_outputs[1], model=model, num_features=300, index2word_set=index2word_set)
    #sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)

    inputs = sys.argv[1]

    line_seg = scoreMatcher.seg_sentence(inputs).strip()  # 这里的返回值是字符串
    line_inputs = line_seg.split(' ')
    print("line_inputs: ", line_inputs)
    inputs = [inputs]
    for line in inputs:
        print("关键词：", HanLP.extractKeyword(line, 4))
        # 自动摘要
        print("摘要句：", HanLP.extractSummary(line, 2))
        line_seg = scoreMatcher.seg_sentence(line).strip()  # 这里的返回值是字符串


    load_model()

    sin_output = []
    avg_vec = []
    rankList = []
    rankDict = {}
    for token in line_inputs:
        #print("word ", token, "with embedding vector: ", model[token])#, embeddings_index[word])
        avg_vec.append(model[token])
    sin_output = np.mean(avg_vec,axis=0)

    
    for sentence, embedding in knowledgeBase.items():
        trans_emb = np.array(embedding, dtype=np.float)
        sim_score = simlarityCalu(sin_output, trans_emb)
        rankList.append(sim_score)
        print("cos outputs: ",sentence,' with ', sim_score)
        rankDict[sentence] = sim_score
    # print("ranked list: ", sorted(rankDict.items()))

    def sorted_dict(container, keys, reverse):
         """返回 keys 的列表,根据container中对应的值排序"""
         aux = [ (container[k], k) for k in keys]
         aux.sort()
         if reverse: aux.reverse()
         return [(k, v) for v, k in aux]
    print("\n")
    order = 0

    for seg in sorted_dict(rankDict,rankDict.keys(),True):
        print("匹配到最可能的Top%d 问题为"% order,seg[0], " 概率：", order,seg[1])
        order += 1

   # model = gensim.models.Word2Vec.load('D:\BaiduNetdiskDownload\sgns.weibo.word\sgns.weibo.word')
    # 主程序
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #sentences = word2vec.Text8Corpus("D:\wiki\11.txt")  
    n_dim=300
     # 训练skip-gram模型; 
    #model = word2vec.Word2Vec(sentences, size=n_dim, min_count=5,sg=1) 
    # 计算两个词的相似度/相关程度
    print("y1")
    print("--------")
    # 寻找对应关系

