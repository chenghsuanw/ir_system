'''
vanilla tf-idf cosine-similairty model
'''

import math

import scipy.sparse
import numpy as np 
from sklearn.preprocessing import normalize

from models.model import Model


class VSM(Model):

	def __init__(self, index_dir_path):

		Model.__init__(self, index_dir_path)

		self.name = "tf-idf-cosine-vsm"

		self.idf_dict = {}

		df = self.get_all_df()

		self.idf = self.compute_idf(df, self.num_docs)


	def compute_idf(self, df, num_docs):

		# idf = log(N / df)

		idf = {}
		for i, v in df.items():
			idf[i] = math.log(num_docs / v)

		return idf 


	def find(self, query, top=1000):

		# query is a string
		# return a list of tuple (entity, score)

		q_dict = self.query2index(query)
		# q_dict: {word_index: freq}

		# add idf weight in query
		q_weight = {w_id: freq * self.idf[w_id] for w_id, freq in q_dict.items()}

		word_indices = [wi for wi in q_dict]
		docs = self.get_docs_by_word_id(word_indices)
		# docs: [(doc_id, {word_index: freq})]

		similarities = np.zeros((len(docs)))
		entities = []

		for n, (doc_id, doc) in enumerate(docs):
			# sq2: compute nomalized term, not a real value, but only compute query words
			sq2 = 0 
			for w_id, freq in doc.items():
				tmp = freq * self.idf[w_id]
				similarities[n] += tmp * q_weight[w_id]
				sq2 += tmp ** 2
			similarities[n] /= np.sqrt(sq2) 
			entities.append(self.i2e[doc_id])

		doc_scores = [(entity, score) for entity, score in zip(entities, similarities)]

		# ranking
		top_k = sorted(doc_scores, key=lambda x:x[1], reverse=True)[:top]

		return top_k
















