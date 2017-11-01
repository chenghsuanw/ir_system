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
parser.add_argument("--output", type=str, default="../db/small_lm", help="output directory path")
parser.add_argument("--chunk_size", type=int, default=10000, help="index chunk size")

FLAGS = parser.parse_args()

if not os.path.exists(FLAGS.output):
	os.makedirs(FLAGS.output)


class DBBuilder(DBBuilderBase):

	# all you need is to implement build_db()

	def build_db(self, D, w2i, entity2i):

		s_time = time.time()

		db_path = os.path.join(FLAGS.output, "db.sqlite")

		conn = sqlite3.connect(db_path)
		db = conn.cursor()

		# to speed up
		db.execute("""PRAGMA synchronous = OFF""")

		# create 3 tables: 
		# doc_word: word probs in a document
	

		db.execute("DROP TABLE IF EXISTS doc_word")
		db.execute("DROP TABLE IF EXISTS corpus_prob")

		db.execute("CREATE TABLE doc_word(doc_id INTEGER, w_id INTEGER, prob REAL)")
		db.execute("CREATE TABLE corpus_prob(w_id INTEGER, prob REAL)")

		# compute corpus_prob

		corpus_prob = np.zeros((len(w2i)), dtype=np.float32)

		for d in D:

			if d["abstract"]:

				doc_id = entity2i[d["entity"]]
				word_indices = [w2i[w] for w in self.clear_string(d["abstract"]).split() if w in w2i]

				word_count = Counter(word_indices)

				for w_id, freq in word_count.items():
					corpus_prob[w_id] += freq 

		# total word count
		total_word_count = int(corpus_prob.sum())

		# normalized to prob
		corpus_prob = corpus_prob / total_word_count

		# insert to db
		db.executemany("INSERT INTO corpus_prob(w_id, prob) VALUES (?, ?)", [(i, float(p)) for i, p in enumerate(corpus_prob)])




		# cache
		cache = []

		for d in D:

			if d["abstract"]:

				doc_id = entity2i[d["entity"]]
				word_indices = [w2i[w] for w in self.clear_string(d["abstract"]).split() if w in w2i]

				word_count = Counter(word_indices)

				# normalize term: sum

				sq2 = 0

				for w_id, freq in word_count.items():
					sq2 += freq 

				for w_id, freq in word_count.items():

					prob = freq / sq2

					cache.append((doc_id, w_id, prob))

				if len(cache) > FLAGS.chunk_size:
					# add to db
					db.executemany("INSERT INTO doc_word(doc_id, w_id, prob) VALUES (?, ?, ?)", cache)
					# flush cache
					cache = []

		if len(cache) > 0:
			db.executemany("INSERT INTO doc_word(doc_id, w_id, prob) VALUES (?, ?, ?)", cache)


		# build index
		logging.info("create index on tables")

		db.execute("CREATE INDEX doc_word_index ON doc_word(w_id)")
		db.execute("CREATE INDEX corpus_prob_index ON corpus_prob(w_id)")

		conn.commit()
		conn.close()

		logging.info("build database time cost: {}".format(time.time() - s_time))

		return total_word_count


if __name__ == "__main__":

	s_time = time.time()

	db_builder = DBBuilder(FLAGS.input, FLAGS.stop_word_path, FLAGS.output, FLAGS.min_count, FLAGS.chunk_size)

	db_builder.build()

	logging.info("total time cost: {}".format(time.time() - s_time))





