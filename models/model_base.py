'''
a model framework
'''
import json
import os 
import pickle
import re
from collections import Counter

import sqlite3


class ModelBase(object):

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

	def find(self, query, top=1000):
		raise NotImplementedError

