# coding=utf-8

import jieba
from collections import Counter
import gensim


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
        sentence_seged = jieba.cut(sentence.strip())
        stopwords = self.stopwordslist('stopwords.txt')  # 这里加载停用词的路径
        outstr = ''
        for word in sentence_seged:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr

    # 计算关键词匹配得分
    def score_compute(self, inputs1, inputs2):
        score = len(inputs1.intersection(inputs2)) / max(len(inputs1), len(inputs2)) 

        return score


if __name__ == '__main__':
    scoreMatcher = keywordMatchScore()
    inputs = open('test_input.txt', 'r') #加载要处理的文件的路径
    outputs = open('test_output.txt', 'w') #加载处理后的文件路径
    char_outputs = []
    line_outputs = []
    for line in inputs:
        line_seg = scoreMatcher.seg_sentence(line)  # 这里的返回值是字符串
        print("lines: ", line_seg)
        char_seg = [line[i] for i in range(len(line))]
        print("chars: ", char_seg)
        line_outputs.append(line_seg.split(' '))
        print("line_seg: ", line_outputs)
        char_outputs.append(char_seg)
        outputs.write(line_seg)
    outputs.close()
    inputs.close()


    inputs1, ner1 = set(char_outputs[0]), set(line_outputs[0])
    inputs2, ner2 = set(char_outputs[1]), set(line_outputs[1])
    score_char = scoreMatcher.score_compute(inputs1, inputs2)
    score_ner = scoreMatcher.score_compute(ner1, ner2)
    score = score_char * (1 / score_ner)

    print("test input: ", inputs1, " and ", inputs2)
    print("with scores:", score, " with ner:", score_ner)

    # WordCount
    with open('test_output.txt', 'r') as fr: #读入已经去除停用词的文件
        data = jieba.cut(fr.read())


    data_count = dict(Counter(data))
 
    print("sentence 1: ", data)
    # print("sentence 1: ", data[1])

    with open('wordCounts.csv', 'w') as fw: #读入存储wordcount的文件路径
        for k,v in data_count.items():
            if k != ' ':
                fw.write('%s,%d\n' % (k, v))