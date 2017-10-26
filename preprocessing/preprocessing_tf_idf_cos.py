import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
handler = logger.handlers[0] # stream handler
formatter = handler.formatter
handler = logging.FileHandler("./log")
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s:%(levelname)s: %(message)s'))
logger.addHandler(handler)


import argparse
import time
import re
import os 
from collections import Counter, defaultdict

import numpy as np 
import progressbar as pb
import sqlite3

from preprocessing_base import DBBuilderBase


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="../datasets/small/DBdoc.json", help="input json file path")
parser.add_argument("--stop_word_path", type=str, default="../stop_words_en.txt", help="english stop words file path")
parser.add_argument("--min_count", type=int, default=5, help="only keep frequency > min_count words")
parser.add_argument("--output", type=str, default="../db/small_tf_idf_cos", help="output directory path")
parser.add_argument("--chunk_size", type=int, default=10000, help="index chunk size")

FLAGS = parser.parse_args()

if not os.path.exists(FLAGS.output):
	os.makedirs(FLAGS.output)


class DBBuilder(DBBuilderBase):

	# all you need is to implement build_db()

	def insert_db(self, db, cache):

		# cache: [(doc_id, w_id, weight)]

		# insert the data in cache to db tables

		# to doc_word table
		db.executemany("INSERT INTO doc_word(doc_id, w_id, weight) VALUES (?, ?, ?)", cache)


	def build_db(self, D, w2i, entity2i):

		s_time = time.time()

		db_path = os.path.join(FLAGS.output, "db.sqlite")

		conn = sqlite3.connect(db_path)
		db = conn.cursor()

		# to speed up
		db.execute("""PRAGMA synchronous = OFF""")

		# create 3 tables: 
		# doc_word: word frequency in a document
		# doc: document word count (document length)

		db.execute("DROP TABLE IF EXISTS doc_word")
		db.execute("DROP TABLE IF EXISTS idf")

		db.execute("CREATE TABLE doc_word(doc_id INTEGER, w_id INTEGER, weight REAL)")
		db.execute("CREATE TABLE idf(w_id INTEGER, weight REAL)")


		# compute all df

		df = np.zeros((len(w2i)), dtype=np.float32)

		for d in D:

			if d["abstract"]:

				doc_id = entity2i[d["entity"]]
				word_indices = [w2i[w] for w in self.clear_string(d["abstract"]).split() if w in w2i]

				word_count = Counter(word_indices)

				for w_id in word_count.keys():
					df[w_id] += 1

		# compute idf
		idf = np.log(len(entity2i) / df)

		# store idf in db                   Note that float(w) because w is numpy.float64 class, not float class
		db.executemany("INSERT INTO idf(w_id, weight) VALUES (?, ?)", [(i, float(w)) for i, w in enumerate(idf)])

		# normalize term
		# sq2 = np.zeros((len(entity2i)), dtype=np.float32)

		# total word count
		total_word_count = 0

		# cache
		cache = []

		for d in D:

			if d["abstract"]:

				doc_id = entity2i[d["entity"]]
				word_indices = [w2i[w] for w in self.clear_string(d["abstract"]).split() if w in w2i]

				word_count = Counter(word_indices)

				# normalize term: l2-norm

				sq2 = 0

				for w_id, freq in word_count.items():
					sq2 += (freq * idf[w_id]) ** 2

				sq2 = np.sqrt(sq2)

				for w_id, freq in word_count.items():

					total_word_count += freq

					# tf-idf 
					# we multiply twice because one idf is from query
					# we multiply here so that we don't need to multiply latter (more fast)
					weight = freq * idf[w_id] * idf[w_id] / sq2

					cache.append((doc_id, w_id, weight))

				if len(cache) > FLAGS.chunk_size:
					# add to db
					self.insert_db(db, cache)
					# flush cache
					cache = []

		if len(cache) > 0:
			self.insert_db(db, cache)


		# build index
		logging.info("create index on tables")

		db.execute("CREATE INDEX doc_word_index ON doc_word(w_id)")
		db.execute("CREATE INDEX idf_index ON idf(w_id)")

		conn.commit()
		conn.close()

		logging.info("build database time cost: {}".format(time.time() - s_time))

		return total_word_count


if __name__ == "__main__":

	s_time = time.time()

	db_builder = DBBuilder(FLAGS.input, FLAGS.stop_word_path, FLAGS.output, FLAGS.min_count, FLAGS.chunk_size)

	db_builder.build()

	logging.info("total time cost: {}".format(time.time() - s_time))





