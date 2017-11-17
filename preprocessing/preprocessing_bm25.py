import json
import re
import pickle
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def main():
	# load stop words
	vectorizer = CountVectorizer(stop_words='english')
	stop_word_set = vectorizer.get_stop_words()
	stop_word_set = stop_word_set.union(set(stopwords.words('english')))


	D_raw = json.load(open("../datasets/small/DBdoc.json", "r"))
	# D: [{'entity':entity_string,'abstract':[word]}]
	D = []
	for i, j in enumerate(D_raw):
		if D_raw[i]['abstract']:
			D_raw[i]['abstract'] = re.sub("[^a-z]", " ", D_raw[i]['abstract'].strip().lower()).split()
			if D_raw[i]['abstract']:
				D.append(D_raw[i])
	with open("../datasets/small/documents","wb") as f:
		pickle.dump(D,f)


	# make each document a counter
	doc_counter_list = []
	for i in range(len(D)):
		c = Counter()
		for w in D[i]['abstract']:
			if not w in stop_word_set:
				c[w] += 1
		doc_counter_list.append(c)
	with open("../datasets/small/doc_counter_list","wb") as f:
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

	with open("../datasets/small/word_appear_doc_dict","wb") as f:
		pickle.dump(word_appear_doc_dict,f)


if __name__ == '__main__':
	main()
