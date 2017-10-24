'''
a model framework
'''
import json
import os 
import pickle
import re
from collections import Counter

import sqlite3


class Model(object):

	def __init__(self, index_dir_path):

		self.name = None

		self.index_dir_path = index_dir_path

		meta_data_path = os.path.join(index_dir_path, "meta_data.json")
		meta_data = json.load(open(meta_data_path))
		self.num_docs = meta_data["num_docs"]
		self.vocab_size = meta_data["vocab_size"]

		vocab_path = os.path.join(index_dir_path, "vocab")
		self.w2i, self.i2w = self.load_vocab(vocab_path)

		doc_id_path = os.path.join(index_dir_path, "doc_ids")
		self.e2i, self.i2e = self.load_doc_ids(doc_id_path)


		db_path = os.path.join(index_dir_path, "db.sqlite")
		conn = sqlite3.connect(db_path)
		db = conn.cursor()
		self.db = db 


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


	def query2index(self, query):

		# query is a string
		# return a dict: {word_index: freq}

		indices = [self.w2i[w] for w in self.clear_string(query).split() if w in self.w2i]

		return dict(Counter(indices))


	def tf_by_word_id_doc_id(self, word_id, doc_id):

		# get term frequency by specific word index and document index
		# avoid to use this because the index of this table, doc_word, is built on only word_id

		# word_id = word index (integer)
		# doc_id: integer

		# return: scalar

		cmd = "SELECT freq FROM doc_word WHERE doc_id = {} AND w_id = {}".format(doc_id, word_id)

		r = self.db.execute(cmd).fetchall()


		if len(r) > 0:
			return r[0][0]
		else:
			return 0

	def tf_by_doc_id(self, doc_id):

		# get all term frequency by specific document index
		# avoid to use this because the index of this table, doc_word, is built on only word_id

		# doc_id: integer

		# return: {word_index:freq}

		cmd = "SELECT w_id, freq FROM doc_word WHERE doc_id = {}".format(doc_id)

		r = self.db.execute(cmd).fetchall()


		doc = {w_id:freq for w_id, freq in r}

		# return all term frequency in this doc
		
		return doc

	def tf_by_word_id(self, word_id):

		# get term frequency in all documents by specific word index

		# return: {doc_id:freq}

		cmd = "SELECT doc_id, freq FROM doc_word WHERE w_id = {}".format(word_id)

		r = self.db.execute(cmd).fetchall()

		inverted_index = {doc_id:freq for doc_id, freq in r}

		return inverted_index

	def get_df(self, word_id):

		# get document frequency by word index

		# return: scalar


		cmd = "SELECT df FROM word WHERE w_id = {}".format(word_id)

		r = self.db.execute(cmd).fetchall()



		if len(r) > 0:
			return r[0][0]
		else:
			return 0

	def get_all_df(self):

		# get all document frequency
		# return: {word_index: freq}

		cmd = "SELECT w_id, df FROM word"
		r = self.db.execute(cmd).fetchall()

		return {k:v for k, v in r}


	def get_docs_by_word_id(self, word_id):

		# get document contain given word index or a list of word indices
		# NOTICE: only get the frequency of given word index in each document, for fast computation

		# input: int or [int]
		# return: [(doc_id, {term: freq}, sq2)]

		if type(word_id) == int:
			word_id = [word_id]

		word_id = [str(w) for w in word_id]

		cmd = "SELECT doc_id, w_id, freq FROM doc_word WHERE w_id in ({})".format(", ".join(word_id))

		r = self.db.execute(cmd).fetchall()

		doc_ids = [rr[0] for rr in r]

		docs = {doc_id: {} for doc_id in doc_ids}
		for doc_id, w_id, freq in r:

			docs[doc_id][w_id] = freq 

		return docs.items()

