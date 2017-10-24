import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
handler = logger.handlers[0] # stream handler
formatter = handler.formatter
handler = logging.FileHandler("./log")
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s:%(levelname)s: %(message)s'))
logger.addHandler(handler)


import argparse
import json
import time
import re
import os 
from collections import Counter, defaultdict

import progressbar as pb

import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="../datasets/small/DBdoc.json", help="input json file path")
parser.add_argument("--stop_words", type=str, default="../stop_words_en.txt", help="english stop words file path")
parser.add_argument("--min_count", type=int, default=5, help="only keep frequency > min_count words")
parser.add_argument("--output", type=str, default="../db/small", help="output directory path")
parser.add_argument("--chunk_size", type=int, default=10000, help="index chunk size")

FLAGS = parser.parse_args()

if not os.path.exists(FLAGS.output):
	os.makedirs(FLAGS.output)


def clear_string(s):

	# input a raw text string
	# output a string with only a-z

	return re.sub("[^a-z]", " ", s.strip().lower())


def build_vocab(D):

	# load stop words

	stop_word_path = FLAGS.stop_words

	stop_words = set()

	with open(stop_word_path, "r") as f:
		for line in f:
			stop_words.add(line.strip())

	# build vocab

	s_time = time.time()

	vocab = Counter()

	for d in D:
		# note that some document has no context
		if d["abstract"]:
			vocab.update([w for w in clear_string(d["abstract"]).split() if w not in stop_words])

	# dump vocab sorted by frequency
	vocab_path = os.path.join(FLAGS.output, "vocab")

	# format: word frequency
	# we only keep words with enough frequency (min_count)

	w2i = {}

	with open(vocab_path, "w") as p:
		for i, (w, freq) in enumerate(vocab.most_common()):
			if freq <= FLAGS.min_count:
				break
			else:
				output = "{} {}".format(w, freq)
				p.write("{}\n".format(output))
				w2i[w] = i 

	# vocab summary
	logging.info("vocab size: {}".format(len(w2i)))
	logging.info("build vocab time cost: {}".format(time.time() - s_time))

	# return word to index mapping (dict)
	return w2i


def build_doc_id(D):

	# because entity has "/", we have to index all documents

	s_time = time.time()

	doc_id_path = os.path.join(FLAGS.output, "doc_ids")

	entity2i = {}

	entities = sorted([d["entity"] for d in D])

	with open(doc_id_path, "w") as p:
		for i, e in enumerate(entities):

			entity2i[e] = i 
			output = e
			p.write("{}\n".format(e))

	# document summary
	logging.info("num docs: {}".format(len(entity2i)))
	logging.info("build doc_id time cost: {}".format(time.time() - s_time))

	return entity2i


def insert_db(db, cache):

	# insert the data in cache to db tables

	# to doc_word table
	db.executemany("INSERT INTO doc_word(doc_id, w_id, freq) VALUES (?, ?, ?)", cache)

	# to doc table, insert word freq and sq2 (insert, because doc are always new)
	cache_tmp = defaultdict(int)
	for doc_id, w_id, freq in cache:
		cache_tmp[doc_id] += freq 
	cache_tmp = [(k, v) for k, v in cache_tmp.items()]
	db.executemany("INSERT INTO doc(doc_id, freq) VALUES (?, ?)", cache_tmp)

	# to word table, update word freq and df
	cache_tmp = defaultdict(int)
	cache_tmp2 = defaultdict(int)
	
	for doc_id, w_id, freq in cache:
		cache_tmp[w_id] += freq 
		cache_tmp2[w_id] += 1
	w_ids = [(w_id,) for w_id in cache_tmp.keys()]
	cache_tmp = [(v, cache_tmp2[k], k) for k, v in cache_tmp.items()]
	db.executemany("INSERT OR IGNORE INTO word(w_id) VALUES (?)", w_ids)
	db.executemany("UPDATE word SET freq = freq + ?, df = df + ? WHERE w_id = ?", cache_tmp)


def build_db(D, w2i, entity2i):

	s_time = time.time()

	db_path = os.path.join(FLAGS.output, "db.sqlite")

	conn = sqlite3.connect(db_path)
	db = conn.cursor()

	# to speed up
	db.execute("""PRAGMA synchronous = OFF""")

	# create 3 tables: 
	# doc_word: word frequency in a document
	# word: global word frequency and document frequency of a word
	# doc: document word count (document length)

	db.execute("DROP TABLE IF EXISTS doc_word")
	db.execute("DROP TABLE IF EXISTS word")
	db.execute("DROP TABLE IF EXISTS doc")

	db.execute("CREATE TABLE doc_word(doc_id INTEGER, w_id INTEGER, freq INTEGER)")
	db.execute("CREATE TABLE word(w_id INTEGER, freq INTEGER DEFAULT 0, df INTEGER DEFAULT 0, CONSTRAINT w_id_unique UNIQUE (w_id))")
	db.execute("CREATE TABLE doc(doc_id INTEGER, freq INTEGER)")

	# total word count
	total_word_count = 0

	# cache
	cache = []

	for d in D:

		if d["abstract"]:

			doc_id = entity2i[d["entity"]]
			word_indices = [w2i[w] for w in clear_string(d["abstract"]).split() if w in w2i]

			word_count = Counter(word_indices)

			for w_id, freq in word_count.items():

				cache.append((doc_id, w_id, freq))

				total_word_count += freq 

				if len(cache) > FLAGS.chunk_size:
					# add to db
					insert_db(db, cache)
					# flush cache
					cache = []

	if len(cache) > 0:
		insert_db(db, cache)

	# build index
	logging.info("create index on tables")

	db.execute("CREATE INDEX doc_word_index ON doc_word(w_id)")
	db.execute("CREATE INDEX word_index ON word(w_id)")
	db.execute("CREATE INDEX doc_index ON doc(doc_id)")

	conn.commit()
	conn.close()

	logging.info("build database time cost: {}".format(time.time() - s_time))

	return total_word_count


def dump_meta_data(D, w2i, entity2i, total_word_count):

	s_time = time.time()

	meta_data = {
		"num_docs": len(entity2i), # number of documents with abstract
		"vocab_size": len(w2i), # total word number
		"total_word_count": total_word_count
	}

	meta_data_path = os.path.join(FLAGS.output, "meta_data.json")

	json.dump(meta_data, open(meta_data_path, "w"))

	logging.info("dump meta data time cost: {}".format(time.time() - s_time))


def main():

	D = json.load(open(FLAGS.input, "r"))
	# D is [{"entity":, "abstract":}]

	w2i = build_vocab(D)

	entity2i = build_doc_id(D)

	total_word_count = build_db(D, w2i, entity2i)

	dump_meta_data(D, w2i, entity2i, total_word_count)


if __name__ == "__main__":

	s_time = time.time()

	main()

	logging.info("total time cost: {}".format(time.time() - s_time))





