'''Preprocessing the data and making it ready for the task of codeswitch prediction, here I am trying to create new features
such as current word itself, repeation of the current word, bigram, trigram, number of times current word happened so far, etc'''
# -*- coding: utf-8 -*-
import os
import pickle
import glob
import matplotlib.pyplot as plt
import numpy as np

file_ = open(os.path.join(r"/Users/Rouzbeh/Google Drive/ESCALES/NLP/Switchpoint", "MetadataInterviews2.txt"), "r")
read_file = file_.read()
inview_split = read_file.split('|||')
# List of all words in the interview file
all_words = []
bigram = []
trigram = []
cur1 = ''
for inview in inview_split:
	this_inview = inview.split('\r\n')
	if this_inview[0] == "":
		for i in range(3,len(this_inview)):
			utter = this_inview[i].split('\t')
			for word in utter:
				features = word.split(',')
				try:
					if features[3] != 'punc':
						if features[4] == 'init' and features[5] == 'init':
							bigram.append('init' + ' ' + features[0])
							trigram.append('init' + ' ' + 'init' + ' ' + features[0])
						elif features[4] != 'init' and features[5] == 'init':
							bigram.append(cur1 + ' ' + features[0])
							trigram.append('init' + ' ' + cur1 + ' ' + features[0])
						elif features[4] != 'init' and features[5] != 'init':
							bigram.append(cur1 + ' ' + features[0])
							trigram.append(cur2 + ' ' + cur1 + ' ' + features[0])
						all_words.append(features[0])
						cur2 = cur1
						cur1 = features[0]
				except IndexError:
					"Hehe"
				
	else:
		for i in range(2,len(this_inview)):
			utter = this_inview[i].split('\t')
			for word in utter:
				features = word.split(',')
				try:
					if features[3] != 'punc':
						if features[4] == 'init' and features[5] == 'init':
							bigram.append('init' + ' ' + features[0])
							trigram.append('init' + ' ' + 'init' + ' ' + features[0])
						elif features[4] != 'init' and features[5] == 'init':
							bigram.append(cur1 + ' ' + features[0])
							trigram.append('init' + ' ' + cur1 + ' ' + features[0])
						elif features[4] != 'init' and features[5] != 'init':
							bigram.append(cur1 + ' ' + features[0])
							trigram.append(cur2 + ' ' + cur1 + ' ' + features[0])
						all_words.append(features[0])
						cur2 = cur1
						cur1 = features[0]
				except IndexError:
					"Hehe"
				
file_.close()


# Unique words
print "Hey1 "
uniq_words = list(set(all_words))
count_words = []
# Counting the number of each words
for word in uniq_words:
	cnt = all_words.count(word)
	count_words.append(cnt)

print len(uniq_words)
print len(count_words)

# Unique Bigrams
print "Hey2 "
uniq_bigrams = list(set(bigram))
count_bi = []
# Counting the number of each words
for word in uniq_bigrams:
	cnt = bigram.count(word)
	count_bi.append(cnt)

print len(uniq_bigrams)
print len(count_bi)


# Unique Tigrams
print "Hey3 "
uniq_trigrams = list(set(trigram))
count_tri = []
# Counting the number of each words
for word in uniq_trigrams:
	cnt = trigram.count(word)
	count_tri.append(cnt)

print len(uniq_trigrams)
print len(count_tri)
pickle.dump([uniq_words,count_words], open("unigram.p", "wb"))
pickle.dump([uniq_bigrams,count_bi], open("bigram.p", "wb"))
pickle.dump([uniq_trigrams,count_tri], open("trigram.p", "wb"))
# pickle.dump( favorite_color, open( "save.p", "wb" ) )
# favorite_color = pickle.load( open( "save.p", "rb" ) )

print trigram[0:10]
print bigram[0:10]
print all_words[0:10]






