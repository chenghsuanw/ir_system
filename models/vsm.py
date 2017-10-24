'''
vanilla tf-idf cosine-similairty model
'''

import math

import scipy.sparse
import numpy as np 
from sklearn.preprocessing import normalize

# from model import Model 
from models.model2 import Model2


class VSM(Model2):

	

	def __init__(self, index_dir_path):

		Model2.__init__(self, index_dir_path)

		self.name = "tf-idf-cosine-vsm"

		self.idf_dict = {}

		self.df = self.get_all_df()

		#print("asd", len(self.df))


		df_vec = scipy.sparse.csr_matrix(([math.log(self.num_docs) - math.log(self.df[i]) for i in range(self.vocab_size) if i in self.df], ([0] * self.vocab_size, [i for i in range(self.vocab_size)])))
		# self.idf_vec = math.log(self.num_docs) - df_vec
		self.idf_vec = df_vec

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
		#print("asd1.3")
		for i, tf in doc.items():
			index.append(i)
			value.append(tf)
			# value.append(tf * self.idf(i))
		#print("asd1.4")
		vec = scipy.sparse.csr_matrix((value, ([0] * len(index), index)), shape=(1, self.vocab_size))
		#print("asd1.5")
		vec = vec.multiply(self.idf_vec)
		#print("asd1.6")

		return vec

	def find(self, query, top=1000):

		# query is a string
		# return a list of tuple (entity, score)

		q_dict = self.query2index(query)
		# q_dict: {word_index: freq}

		q_vec = self.doc2bow(q_dict)
		# shape: (1, V)

		word_indices = [wi for wi in q_dict]

		##print("asd0")
		docs = self.get_docs_by_word_id(word_indices)
		# [(doc_id, {word_index: freq})]

		#print("5")
		entities = []
		doc_vecs = []
		##print("asd1")
		for doc_id, doc, sq2 in docs:
			#print("5.1")
			vec = self.doc2bow(doc) / sq2
			#print("5.2")

			entities.append(self.i2e[doc_id])
			doc_vecs.append(vec)

		# ##print("asd1.5")
		doc_vecs = scipy.sparse.vstack(doc_vecs)
		#print("6")
		# cosine similarity

		# normalize (we only do it on doc vecs)
		##print("asd2")
		# doc_vecs = normalize(doc_vecs, norm='l2', axis=1)
		# shape: (n, V)

		# #print("7")

		# compute similarity
		##print("asd3")
		similarities = (doc_vecs * q_vec.T).toarray()[:, 0]
		# shape should be (n,)

		#print("8")

		doc_scores = [(entity, score) for entity, score in zip(entities, similarities)]

		# ranking
		##print("asd4")
		top_k = sorted(doc_scores, key=lambda x:x[1], reverse=True)[:top]

		##print("asd5")

		##print("dhit", self.doc_hit)
		##print("whit", self.word_hit)
		return top_k








		

















