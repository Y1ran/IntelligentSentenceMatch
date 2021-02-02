
# coding=utf-8

import jieba
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

if __name__ == '__main__':
	"""
	在这个例子里，learn_kernel就相当于cos similarity，
	因为sklearn.feature_extraction.text.TfidfVectorizer本身得到的就是归一化后的向量，
	这样cosine_similarity就相当于linear_kernel
	"""
	twenty = fetch_20newsgroups()
	tfidf = TfidfVectorizer().fit_transform(twenty.data)

	print(" tfidf[0:1] is : ",  tfidf[0:1])

	#使用scikit-learn包中已经定义好的计算规则
	from sklearn.metrics.pairwise import linear_kernel
	cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
	print(" cosine_similarities is : ",  cosine_similarities)

	related_docs_indices = cosine_similarities.argsort()[:-5:-1]
	print(related_docs_indices)
	#array([    0,   958, 10576,  3277])
	print(cosine_similarities[related_docs_indices])
	#array([ 1.        ,  0.54967926,  0.32902194,  0.2825788 ])
	#第一个结果用于检查，这是query本身，相似度为1