# coding=utf-8

import jieba
from collections import Counter


class LeveinAlgoScore():
    #编辑距离函数代码
    def __init__(self):
        self.listpath = None
        self.sentence = None

    # 创建停用词列表
    def stopwordslist(self,listpath):
	    stopwords = [line.strip() for line in open(listpath, encoding='GBK').readlines()]
	    return stopwords


    # 对句子进行分词
    def seg_sentence(self,sentence):
        stopwords = self.stopwordslist('stopwords.txt')  # 这里加载停用词的路径
        outstr = ''
        for word in sentence:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr

    def minDistance(self, word1, word2):
        if not word1:
            return len(word2 or '') or 0

        if not word2:
            return len(word1 or '') or 0

        size1 = len(word1)
        size2 = len(word2)

        last = 0
        tmp = list(range(size2 + 1))
        value = None

        for i in range(size1):
            tmp[0] = i + 1
            last = i
            # print word1[i], last, tmp
            for j in range(size2):
                if word1[i] == word2[j]:
                    value = last
                else:
                    value = 1 + min(last, tmp[j], tmp[j + 1])
                    # print(last, tmp[j], tmp[j + 1], value)
                last = tmp[j+1]
                tmp[j+1] = value
            # print tmp
        return value


if __name__ == '__main__':
    scoreMatcher = LeveinAlgoScore()
    inputs = open('test_input.txt', 'r') #加载要处理的文件的路径
    outputs = open('test_output.txt', 'w') #加载处理后的文件路径
    char_outputs = []
    line_outputs = []
    for line in inputs:
        output = scoreMatcher.seg_sentence(line)
        line_outputs.append(output)   # 这里的返回值是字符串

    print("the output lines: ", line_outputs)
    score = scoreMatcher.minDistance(line_outputs[0], line_outputs[1]) / max(len(line_outputs[0]), len(line_outputs[1]))
    print("the final scores: ", score)