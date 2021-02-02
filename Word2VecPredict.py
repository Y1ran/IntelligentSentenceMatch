# coding=GBK

from pyhanlp import *
import traceback
import warnings
import numpy as np
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
from gensim.models import KeyedVectors
from scipy import spatial
from flask import request, Flask
from Word2VecMatch import Word2VecTester, keywordMatchScore,simlarityCalu

app = Flask(__name__)



@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":  #what you want to do with frozen model goes here
        scoreMatcher = keywordMatchScore()
        inputs = open('test_input.txt', 'r') #加载要处理的文件的路径
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
            avg_embedding = np.mean(avg_vec,axis=1)
            avg_embedding_test = np.mean(avg_vec,axis=0)
            print("average embedding1: ", avg_embedding)
            print("average embedding2: ", avg_embedding_test)
            cos_input[''.join(sentence)] = avg_embedding_test
            cos_output.append(avg_embedding_test)

        #index2word_set = set(model.wv.index2word)

        #s1_afv = avg_feature_vector(line_outputs[0], model=model, num_features=300, index2word_set=index2word_set)
        #s2_afv = avg_feature_vector(line_outputs[1], model=model, num_features=300, index2word_set=index2word_set)
        #sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)

        for i in range(len(cos_output)):
            for j in range(i):
                print("cos outputs: ",i, ' and ', j,' with ', simlarityCalu(cos_output[i], cos_output[j]))



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

if __name__ == '__main__':
    print(("* given outpus by model and Flask starting server..."))
    predict()