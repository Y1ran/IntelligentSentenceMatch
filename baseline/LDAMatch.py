import codecs
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import corpora, models
import numpy as np
import jieba
import tutorials.data_util as du

def get_dict():
    train = []
    fp = codecs.open('../xx.txt', 'r', encoding='utf8')#文本文件，输入需要提取主题的文档
    stopwords = ['的', '吗', '我', '会', '使用', '跟', '了', '有', '什么', '这个', '下', '或者', '能', '要', '怎么', '呢', '吧', '都']#取出停用词
    for line in fp:
        line = list(jieba.cut(line))
        train.append([w for w in line if w not in stopwords])

    dictionary = Dictionary(train)
    return dictionary,train
def train_model():
    dictionary=get_dict()[0]
    train=get_dict()[1]
    corpus = [ dictionary.doc2bow(text) for text in train ]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=7)
    #模型的保存/ 加载
    lda.save('test_lda.model')
#计算两个文档的相似度
def lda_sim(s1,s2):
    lda = models.ldamodel.LdaModel.load('test_lda.model')
    test_doc = list(jieba.cut(s1))  # 新文档进行分词
    dictionary=get_dict()[0]
    doc_bow = dictionary.doc2bow(test_doc)  # 文档转换成bow
    doc_lda = lda[doc_bow]  # 得到新文档的主题分布
    # 输出新文档的主题分布
    # print(doc_lda)
    list_doc1 = [i[1] for i in doc_lda]
    # print('list_doc1',list_doc1)

    test_doc2 = list(jieba.cut(s2))  # 新文档进行分词
    doc_bow2 = dictionary.doc2bow(test_doc2)  # 文档转换成bow
    doc_lda2 = lda[doc_bow2]  # 得到新文档的主题分布
    # 输出新文档的主题分布
    # print(doc_lda)
    list_doc2 = [i[1] for i in doc_lda2]
    # print('list_doc2',list_doc2)
    try:
        sim = np.dot(list_doc1, list_doc2) / (np.linalg.norm(list_doc1) * np.linalg.norm(list_doc2))
    except ValueError:
        sim=0
    #得到文档之间的相似度，越大表示越相近
    return sim