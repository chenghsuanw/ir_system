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
import progressbar as pb
import time



parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="../datasets/small/DBdoc.json", help="input json file path")
parser.add_argument("--output", type=str, help="output directory path")


args = parser.parse_args()

