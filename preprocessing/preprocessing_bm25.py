import json
import re
import pickle
import nltk
import sys
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

input_file = sys.argv[1]
documents_file = sys.argv[2]
doc_counter_list_file = sys.argv[3]
word_appear_doc_dict_file = sys.argv[4]

def main():
	# load stop words
	vectorizer = CountVectorizer(stop_words='english')
	stop_word_set = vectorizer.get_stop_words()
	stop_word_set = stop_word_set.union(set(stopwords.words('english')))


	D_raw = json.load(open(input_file, "r"))
	# D: [{'entity':entity_string,'abstract':[word]}]
	D = []
	for i, j in enumerate(D_raw):
		if D_raw[i]['abstract']:
			D_raw[i]['abstract'] = re.sub("[^a-z]", " ", D_raw[i]['abstract'].strip().lower()).split()
			if D_raw[i]['abstract']:
				D.append(D_raw[i])
	with open(documents_file,"wb") as f:
		pickle.dump(D,f)


	# make each document a counter
	doc_counter_list = []
	for i in range(len(D)):
		c = Counter()
		for w in D[i]['abstract']:
			if not w in stop_word_set:
				c[w] += 1
		doc_counter_list.append(c)
	with open(doc_counter_list_file,"wb") as f:
		pickle.dump(doc_counter_list,f)


	# for each word, record which documents it appears
	# {'word': [document]}
	word_appear_doc_dict = dict()
	for d in range(len(D)):
		for word in D[d]['abstract']:
			if not word in stop_word_set:
				if word in word_appear_doc_dict:
					if not d in word_appear_doc_dict[word]:
						word_appear_doc_dict[word].append(d)
				else:
					word_appear_doc_dict[word] = [d]

	with open(word_appear_doc_dict_file,"wb") as f:
		pickle.dump(word_appear_doc_dict,f)


if __name__ == '__main__':
	main()
