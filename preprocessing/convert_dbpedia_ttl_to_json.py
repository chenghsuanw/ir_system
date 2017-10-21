import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
handler = logger.handlers[0] # stream handler
formatter = handler.formatter
handler = logging.FileHandler("./log")
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s:%(levelname)s: %(message)s'))
logger.addHandler(handler)


import rdflib
import argparse
import json
import progressbar as pb
import time


parser = argparse.ArgumentParser()
# long_abstracts_en.ttl wget and extract from http://downloads.dbpedia.org/2015-04/core-i18n/en/long-abstracts_en.ttl.bz2
parser.add_argument("--input", type=str, default="/tmp/gdoggg2032/long_abstracts_en.ttl", help="input ttl file path")
parser.add_argument("--output", type=str, help="output json file path")


args = parser.parse_args()

# record processing time
s_time = time.time()


logging.info("start processing...")

g = rdflib.Graph()


logging.info("read data from: {}".format(args.input))
# this costs long time
g.parse(args.input, format="ttl")


maxval = len(g)
pbar = pb.ProgressBar(
	widgets=[
	"[Processing]",
	pb.FileTransferSpeed(unit="lines"), 
	pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()
	], 
	maxval=maxval
	).start()

data = []

for n, (resource, category, abstract) in enumerate(g):
	entity = str(resource)[28:]
	abstract = str(abstract)

	data.append({"entity": entity, "abstract":abstract})

	pbar.update(n)

pbar.finish()

logging.info("dump to file: {}".format(args.output))
json.dump(data, open(args.output, "w"))

logging.info("total time cost: {}".format(time.time() - s_time))

