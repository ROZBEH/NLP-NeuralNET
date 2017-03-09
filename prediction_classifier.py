'''Here I am going to have some sort of classifier for classifying each point as switch or not a switch'''

# -*- coding: utf-8 -*-
import os
import pickle
import glob
import matplotlib.pyplot as plt
import numpy as np
from bin import *
# Loading the previous ready to use unigram, bigram, and trigrams
[uniq_words,count_words] = pickle.load( open( "unigram.p", "rb" ) )
[uniq_bigrams,count_bi] = pickle.load( open( "bigram.p", "rb" ) )
[uniq_trigrams,count_tri] = pickle.load( open( "trigram.p", "rb" ) )
A  = ['en','sw', 'init', 'mixed','other','same','diff','punc']
file_ = open(os.path.join(r"/Users/Rouzbeh/Google Drive/ESCALES/NLP/Switchpoint", "MetadataInterviews2.txt"), "r")
read_file = file_.read()
inview_split = read_file.split('|||')
# List of all words in the interview file
X = []
y = []

cur1 = ''
for inview in inview_split:
	this_inview = inview.split('\r\n')
	if this_inview[0] == "":
		for i in range(3,len(this_inview)):
			utter = this_inview[i].split('\t')
			for word in utter:
				features = word.split(',')
				x_vec = []
				try:
					if features[3] != 'punc':
						if features[4] == 'init' and features[5] == 'init':
							bigram = 'init' + ' ' + features[0]
							trigram = 'init' + ' ' + 'init' + ' ' + features[0]
						elif features[4] != 'init' and features[5] == 'init':
							bigram = cur1 + ' ' + features[0]
							trigram = 'init' + ' ' + cur1 + ' ' + features[0]
						elif features[4] != 'init' and features[5] != 'init':
							bigram = cur1 + ' ' + features[0]
							trigram = cur2 + ' ' + cur1 + ' ' + features[0]
						cur2 = cur1
						cur1 = features[0]
						# adding unigram, bigram and trigram features
						B1 = uniq_words.index(features[0])
						B2 = uniq_bigrams.index(bigram)
						B3 = uniq_trigrams.index(trigram)
						if count_words[B1] >= 5:
							x_vec.append(1)
							x_vec.append(count_words[B1])
						else:
							x_vec.append(0)
							x_vec.append(count_words[B1])
						if count_bi[B2] >= 2:
							x_vec.append(1)
							x_vec.append(count_bi[B2])
						else:
							x_vec.append(0)
							x_vec.append(count_bi[B2])
						if count_tri[B3] >= 2:
							x_vec.append(1)
							x_vec.append(count_tri[B3])
						else:
							x_vec.append(0)
							x_vec.append(count_tri[B3])
						x_vec.append(A.index(features[3]))
						x_vec.append(A.index(features[4]))
						x_vec.append(A.index(features[5]))
						x_vec.append(A.index(features[6]))
						x_vec.append(A.index(features[7]))
						x_vec.append(int(features[8]))
						x_vec.append(int(features[9]))
						A1 = round(float(features[10]), 2)
						A2 = round(float(features[11]), 2)
						A3 = round(float(features[12]), 2)
						x_vec.append(bin_range(A1,8.95,1))
						x_vec.append(bin_range(A2,8.95,0))
						x_vec.append(bin_range(A3,1,0))
						x_vec.append(int(float(features[13])))
						y.append(int(float(features[14])))
						X.append(x_vec)
				except IndexError:
					"Hehe"
				
	else:
		for i in range(2,len(this_inview)):
			utter = this_inview[i].split('\t')
			for word in utter:
				features = word.split(',')
				x_vec = []
				try:
					if features[3] != 'punc':
						if features[4] == 'init' and features[5] == 'init':
							bigram = 'init' + ' ' + features[0]
							trigram = 'init' + ' ' + 'init' + ' ' + features[0]
						elif features[4] != 'init' and features[5] == 'init':
							bigram = cur1 + ' ' + features[0]
							trigram = 'init' + ' ' + cur1 + ' ' + features[0]
						elif features[4] != 'init' and features[5] != 'init':
							bigram = cur1 + ' ' + features[0]
							trigram = cur2 + ' ' + cur1 + ' ' + features[0]
						cur2 = cur1
						cur1 = features[0]
						# adding unigram, bigram and trigram features
						B1 = uniq_words.index(features[0])
						B2 = uniq_bigrams.index(bigram)
						B3 = uniq_trigrams.index(trigram)
						if count_words[B1] >= 5:
							x_vec.append(1)
							x_vec.append(count_words[B1])
						else:
							x_vec.append(0)
							x_vec.append(count_words[B1])
						if count_bi[B2] >= 2:
							x_vec.append(1)
							x_vec.append(count_bi[B2])
						else:
							x_vec.append(0)
							x_vec.append(count_bi[B2])
						if count_tri[B3] >= 2:
							x_vec.append(1)
							x_vec.append(count_tri[B3])
						else:
							x_vec.append(0)
							x_vec.append(count_tri[B3])
						x_vec.append(A.index(features[3]))
						x_vec.append(A.index(features[4]))
						x_vec.append(A.index(features[5]))
						x_vec.append(A.index(features[6]))
						x_vec.append(A.index(features[7]))
						x_vec.append(int(features[8]))
						x_vec.append(int(features[9]))
						A1 = round(float(features[10]), 2)
						A2 = round(float(features[11]), 2)
						A3 = round(float(features[12]), 2)
						x_vec.append(bin_range(A1,8.95,1))
						x_vec.append(bin_range(A2,8.95,0))
						x_vec.append(bin_range(A3,1,0))
						x_vec.append(int(float(features[13])))
						y.append(int(float(features[14])))
						X.append(x_vec)
				except IndexError:
					"Hehe"
				
file_.close()

print len(X)
print len(y)
pickle.dump(X, open("X_feature_all_tag_count.p", "wb"))
pickle.dump(y, open("y_label_all_tag_count.p", "wb"))
print X[0:10]
# most frequesnt unigrams, bigrams, trigrams or someothertypes of measurements or the frequesncy of repetition of each word 
# also might be helpful for us