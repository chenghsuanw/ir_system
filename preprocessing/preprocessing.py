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
from collections import Counter
import pickle


import progressbar as pb



parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="../datasets/small/DBdoc.json", help="input json file path")
parser.add_argument("--output", type=str, default="../indices/small", help="output directory path")


FLAGS = parser.parse_args()

if not os.path.exists(FLAGS.output):
	os.makedirs(FLAGS.output)



def clear_string(s):
	# input a raw text string
	# output a string with only a-z

	return re.sub("[^a-z]", " ", s.strip().lower())



def build_vocab(D):

	# build vocab

	s_time = time.time()

	vocab = Counter()

	for d in D:
		# note that some document has no context
		if d["abstract"]:
			vocab.update(clear_string(d["abstract"]).split())

	vocab_size = len(vocab)

	# dump vocab sorted by frequency
	vocab_path = os.path.join(FLAGS.output, "vocab")


	# format: word frequency
	# note that here we don't cut off any word even its frequency is very low

	w2i = {}

	with open(vocab_path, "w") as p:
		for i, (w, freq) in enumerate(vocab.most_common()):
			output = "{} {}".format(w, freq)
			p.write("{}\n".format(output))
			w2i[w] = i 

	# vocab summary
	logging.info("vocab size: {}".format(vocab_size))
	logging.info("build vocab time cost: {}".format(time.time() - s_time))

	# return word to index mapping (dict)
	return w2i


def build_doc_id(D):

	# because entity has /, we have to index all documents

	s_time = time.time()

	doc_id_path = os.path.join(FLAGS.output, "doc_ids")

	entity2i = {}

	entities = sorted([d["entity"] for d in D])

	with open(doc_id_path, "w") as p:
		for i, e in enumerate(entities):

			entity2i[e] = i 
			output = e
			p.write("{}\n".format(e))


	return entity2i





def dump_documents(D, w2i, entity2i):

	# dump documents

	s_time = time.time()

	docs_dir_path = os.path.join(FLAGS.output, "docs")

	if not os.path.exists(docs_dir_path):
		os.makedirs(docs_dir_path)

	for d in D:

		# note that some document has no context
		if d["abstract"]:
			
			doc_id = entity2i[d["entity"]]

			doc_path = os.path.join(docs_dir_path, str(doc_id))

			# convert words to index
			indices = [w2i[w] for w in clear_string(d["abstract"]).split() if w in w2i]

			# we use dict to store a doc
			# format: {word_index: freq}
			doc = dict(Counter(indices))

			# save in pickle format

			pickle.dump(doc, open(doc_path, "wb"))

	logging.info("dump documents time cost: {}".format(time.time() - s_time))



def update_inverted_index(inverted_index, index_dir_path):

	logging.info("updating inverted_index")

	for word_index, index in inverted_index.items():

		word_index_path = os.path.join(index_dir_path, str(word_index))

		if os.path.isfile(word_index_path):

			d = pickle.load(open(word_index_path, "rb"))

			d["df"] += index["df"]
			d["tf"].update(index["tf"])

			pickle.dump(d, open(word_index_path, "wb"))

		else:

			pickle.dump(index, open(word_index_path, "wb"))




def dump_inverted_index(D, w2i, entity2i, chunk_size=100000):


	s_time = time.time()


	index_dir_path = os.path.join(FLAGS.output, "inverted_index")

	if not os.path.exists(index_dir_path):
		os.makedirs(index_dir_path)


	# save df, tf
	# format: {"df": int, "tf":{doc_id: freq}}

	inverted_index = {}

	num_doc = len(D)

	for n, d in enumerate(D):

		if d["abstract"]:

			doc_id = entity2i[d["entity"]]
			indices = [w2i[w] for w in clear_string(d["abstract"]).split() if w in w2i]

			for i in indices:

				if i not in inverted_index:
					inverted_index[i] = {}
					inverted_index[i]["df"] = 0
					inverted_index[i]["tf"] = {}

				inverted_index[i]["df"] += 1 

				if doc_id not in inverted_index[i]["tf"]:
					inverted_index[i]["tf"][doc_id] = 0

				inverted_index[i]["tf"][doc_id] += 1


		if n % chunk_size == 0:
			logging.info("processing {}/{}".format(n, num_doc))
			update_inverted_index(inverted_index, index_dir_path)
			inverted_index = {}

	if len(inverted_index) > 0:
		update_inverted_index(inverted_index, index_dir_path)


	logging.info("dump inverted index time cost: {}".format(time.time() - s_time))





def main():

	D = json.load(open(FLAGS.input, "r"))
	# D is [{"entity":, "abstract":}]

	w2i = build_vocab(D)

	entity2i = build_doc_id(D)

	dump_documents(D, w2i, entity2i)

	dump_inverted_index(D, w2i, entity2i)







if __name__ == "__main__":

	s_time = time.time()

	main()

	logging.info("total time cost: {}".format(time.time() - s_time))




