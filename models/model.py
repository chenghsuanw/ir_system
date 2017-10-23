'''
a model framework
'''
import json
import os 
import pickle
import re
from collections import Counter







class Model(object):

	def __init__(self, index_dir_path):

		self.index_dir_path = index_dir_path
		self.doc_index_dir_path = os.path.join(self.index_dir_path, "docs")
		self.inverted_index_dir_path = os.path.join(self.index_dir_path, "inverted_index")

		meta_data_path = os.path.join(index_dir_path, "meta_data.json")
		meta_data = json.load(open(meta_data_path))
		self.num_docs = meta_data["num_docs"]
		self.vocab_size = meta_data["vocab_size"]
		self.word_index_chunk_size = meta_data["word_index_chunk_size"]
		self.document_index_chunk_size = meta_data["document_index_chunk_size"]

		vocab_path = os.path.join(index_dir_path, "vocab")
		self.w2i, self.i2w = self.load_vocab(vocab_path)

		doc_id_path = os.path.join(index_dir_path, "doc_ids")
		self.e2i, self.i2e = self.load_doc_ids(doc_id_path)

		# self.df = self.load_df()

	def clear_string(self, s):
		# input a raw text string
		# output a string with only a-z

		return re.sub("[^a-z]", " ", s.strip().lower())

	def load_vocab(self, vocab_path):

		w2i = {}
		i2w = {}

		with open(vocab_path, "r") as f:
			for index, line in enumerate(f):
				ll = line.strip().split()
				word = ll[0]
				freq = int(ll[1])
				w2i[word] = index 
				i2w[index] = word 

		return w2i, i2w

	def load_doc_ids(self, doc_id_path):

		e2i = {}
		i2e = {}

		with open(doc_id_path, "r") as f:
			for index, line in enumerate(f):
				entity = line.strip()
				e2i[entity] = index 
				i2e[index] = entity

		return e2i, i2e


	def load_df(self):

		df = {}

		for i in self.i2w:
			df[i] = self.get_df(i)

		return df 




	def query2index(self, query):

		# query is a string
		# return a dict: {word_index: freq}

		indices = [self.w2i[w] for w in self.clear_string(query).split() if w in self.w2i]

		return dict(Counter(indices))


	def tf_by_word_id_doc_id(self, word_id, doc_id):

		# get term frequency by specific word index and document index

		# word_id = word index (integer)
		# doc_id: integer

		# return: scaler

		# from document index
		document_path = os.path.join(self.doc_index_dir_path, str(doc_id % self.document_index_chunk_size))

		doc = pickle.load(open(document_path, "rb"))[doc_id]

		return doc[word_id] if word_id in doc else 0

	def tf_by_doc_id(self, doc_id):

		# get all term frequency by specific document index

		# doc_id: integer

		# return: {word_index:freq}

		# from document index
		document_path = os.path.join(self.doc_index_dir_path, str(doc_id % self.document_index_chunk_size))

		doc = pickle.load(open(document_path, "rb"))[doc_id]

		# return all term frequency in this doc
		return doc

	def tf_by_word_id(self, word_id):

		# get term frequency in all documents by specific word index

		# return: {doc_id:freq}

		word_path = os.path.join(self.inverted_index_dir_path, str(word_id % self.word_index_chunk_size))

		inverted_index = pickle.load(open(word_path, "rb"))[word_id]

		return inverted_index

	def get_df(self, word_id):

		# get document frequency by word index

		# return: scalar

		word_path = os.path.join(self.inverted_index_dir_path, str(word_id % self.word_index_chunk_size))

		inverted_index = pickle.load(open(word_path, "rb"))[word_id]

		return len(inverted_index)

	def get_docs_by_word_id(self, word_id):

		# get document bag of word dictionaries by given word index or a list of word indices

		# input: int or [int]
		# return: [(doc_id, {term: freq})]

		if type(word_id) == int:
			word_id = [word_id]


		doc_ids = set()

		for wi in word_id:
			tf = self.tf_by_word_id(wi)
			# tf: {doc_id: freq}
			doc_ids.update(tf.keys())

		docs = []

		for doc_id in doc_ids:

			doc = self.tf_by_doc_id(doc_id)

			docs.append((doc_id, doc))


		return docs 







