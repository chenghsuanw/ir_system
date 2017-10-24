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

import pandas as pd 
import progressbar as pb


parser = argparse.ArgumentParser()
parser.add_argument("--query", type=str, default="./datasets/small/queries-v2.txt", help="input query file path")
parser.add_argument("--output", type=str, default="./result.run", help="output format file path")
parser.add_argument("--index", type=str, default="./db/small", help="index directory path")
parser.add_argument("--model", type=str, default="vsm", help="model type: [vsm, ]")
parser.add_argument("--top", type=int, default=1000, help="output top k ranked list")

FLAGS = parser.parse_args()


def select_model(model_name):

	model = None

	if model_name == "vsm":
		from models.vsm import VSM 
		model = VSM(FLAGS.index)

	return model 


def retrieval(model, query):

	# input: query, including query id and context
	# return: [{result format}]

	raw_result = model.find(query.query, top=FLAGS.top)
	# result = [(entity, score)]

	# result format: "query_id" Q0 <dbpedia:"entity"> "rank_from_1" "score" "model_name"
	# e.g., INEX_LD-20120111 Q0 <dbpedia:Vietnam_movie> 1 8.25869664 bm25

	result = []

	for n, (entity, score) in enumerate(raw_result):

		result_dict = {
			"query_id": query.query_id,
			"Q0": "Q0",
			"entity": "<dbpedia:{}".format(entity),
			"rank": n + 1,
			"score": score,
			"model_name": model.name
		}

		result.append(result_dict)

	return result 


def main():

	# load queries
	logging.info("load queries from: {}".format(FLAGS.query))
	queries = pd.read_csv(FLAGS.query, sep="\t", header=None)
	queries = queries.rename(columns={0: "query_id", 1: "query"})
	# queries: two columns: query_id, query

	# select model
	model = select_model(FLAGS.model)
	logging.info("model [{}] initialized".format(model.name))

	# retrieval
	results = []

	maxval = len(queries)
	pbar = pb.ProgressBar(
		widgets=[
		"[Retrieval]",
		pb.FileTransferSpeed(unit="queries"), 
		pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()
		], 
		maxval=maxval
		).start()

	for n, query in queries.iterrows():

		result = retrieval(model, query)

		results.extend(result)

		pbar.update(n)

	pbar.finish()

	# dump result

	result_df = pd.DataFrame(results)

	columns = ["query_id", "Q0", "entity", "rank", "score", "model_name"]

	logging.info("dump result to: {}".format(FLAGS.output))
	result_df.to_csv(FLAGS.output, columns=columns, index=None, header=None, sep=" ")


if __name__ == "__main__":

	s_time = time.time()

	main()

	logging.info("total time cost: {}".format(time.time() - s_time))









