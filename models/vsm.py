'''
vanilla tf-idf cosine-similairty model
'''

import math

import scipy.sparse
import numpy as np 
from sklearn.preprocessing import normalize

from model import Model 


class VSM(Model):

	

	def __init__(self, index_dir_path):

		Model.__init__(self, index_dir_path)

		self.idf_dict = {}

	def compute_idf(self, df, num_docs):

		# idf = log(N / df)

		idf = {}
		for i, v in df.items():
			idf[i] = math.log(num_docs / v)

		return idf 

	
	def idf(self, word_index):

		# idf = log(N / df)

		if word_index in self.idf_dict:
			return self.idf_dict[word_index]
		else:
			self.idf_dict[word_index] = math.log(self.num_docs / self.get_df(word_index))
			return self.idf_dict[word_index]

	def doc2bow(self, doc):

		# doc is a dict: {word_index:freq}
		# return a bow scipy sparse csr matrix

		value = []
		index = []

		for i, tf in doc.items():
			index.append(i)
			value.append(tf * self.idf(i))

		print(self.vocab_size)
		vec = scipy.sparse.csr_matrix((value, ([0] * len(index), index)), shape=(1, self.vocab_size))

		return vec

	def find(self, query, top=1000):

		# query is a string
		# return a list of tuple (entity, score)

		q_dict = self.query2index(query)
		# q_dict: {word_index: freq}

		q_vec = self.doc2bow(q_dict)
		# shape: (1, V)

		word_indices = [wi for wi in q_dict]

		docs = self.get_docs_by_word_id(word_indices)
		# [(doc_id, {word_index: freq})]

		entities = []
		doc_vecs = []

		for doc_id, doc in docs:
			vec = self.doc2bow(doc)

			entities.append(self.i2e[doc_id])
			doc_vecs.append(vec)

		doc_vecs = scipy.sparse.vstack(doc_vecs)

		# cosine similarity

		# normalize (we only do it on doc vecs)
		
		doc_vecs = normalize(doc_vecs, norm='l2', axis=1)
		# shape: (n, V)

		# compute similarity

		similarities = (doc_vecs * q_vec.T).toarray()[:, 0]
		# shape should be (n,)

		doc_scores = [(entity, score) for entity, score in zip(entities, similarities)]

		# ranking

		top_k = sorted(doc_scores, key=lambda x:x[1], reverse=True)[:top]

		return top_k








		

















