import json
import re
import pickle
import math
import numpy as np
import operator

def main():
	# D : [{'entity':'string','abstract':'[word]'}]
	with open("../datasets/small/documents","rb") as f:
		D = pickle.load(f)
		
	# doc_counter_list : [doc_counter]
	with open("../datasets/small/doc_counter_list","rb") as f:
		doc_counter_list = pickle.load(f)

	# word_appear_doc_dict : {'word': [documents]}
	with open("../datasets/small/word_appear_doc_dict","rb") as f:
		word_appear_doc_dict = pickle.load(f)

	doc_count = len(D)

	total_doc_length = 0
	for i in range(doc_count):
		total_doc_length += len(D[i]['abstract'])
	ave_doc_length = float(total_doc_length)/doc_count

	# Q is a list : [query_id, [query_word]]
	Q = []
	f = open("../datasets/small/queries-v2.txt")
	line = f.readline()	
	while line:
		Q.append(line.split('\t'))
		Q[-1][1] = re.sub("[^a-z]", " ", Q[-1][1].strip().lower()).split()
		line = f.readline()
	f.close()

	K1 = 1
	b = 0.75

	f = open("result.txt","w")
	for q in range(len(Q)):
		score = np.zeros(doc_count)
		for term in Q[q][1]:
			if term in word_appear_doc_dict:
				appear_doc = word_appear_doc_dict[term]
				for i in appear_doc:
					B_ij = (K1+1)*doc_counter_list[i][term] / (K1*((1-b)+b*(len(D[i]['abstract'])/ave_doc_length))+doc_counter_list[i][term])
					score[i] += B_ij*math.log2((doc_count-len(appear_doc)+0.5)/(len(appear_doc)+0.5))
		doc_score = tuple(zip(range(doc_count), score)) 
		result = sorted(doc_score, key=operator.itemgetter(1), reverse=True)
		for i in range(1000):
			f.write("{}\tQ0\t<dbpedia:{}>\t{}\t{}\tSTANDARD\n".format(Q[q][0], D[result[i][0]]['entity'], i+1, result[i][1]))

	f.close()
if __name__ == '__main__':
	main()