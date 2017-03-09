'''Here I am going to work with the JamiForum Data and extract some of the interesting stats about this Forum'''
#@author rouzbeh
# -*- coding: utf-8 -*-
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import collections

file_ = open("Feat_New.txt", "r")
read_file = file_.read()
JF_split = read_file.split('\n')
num_words = 0
num_post = len(JF_split)

'''Each feature vector consists of the following features, 
"lang","lang-1","lang-2","same/diff","same/diff",
"sameLang","diffLang","sameLangLog","diffLangLog","sameLangRatio","preCoSw","CoSwPoint", "word","POS","lemma"
Also the numbers inside the feature vector for language labels are as follows:
0 : swahili, 1 : english, 2 : mixed, 3 : other, -1 : init, 4 : pun
'''
switch_lang = []
for post in JF_split:
	post_split = post.split('\t')[:-1]
	num_words += len(post_split)
	for ind,word_feat in enumerate(post_split):
		features = word_feat.split(',')
		# because I have different types of punctuations so I use all of these numbers for features[0]
		if features[-4] == '1' and features[0] != '4' and features[0] != '5' and features[0] != '6' and features[0] != '7': 
			while (post_split[ind + 1].split(',')[0]=='4' or post_split[ind + 1].split(',')[0]=='5' or post_split[ind + 1].split(',')[0]=='6' or post_split[ind + 1].split(',')[0]=='7'):
				ind += 1
			switch_lang.append(features[1] + features[0] + post_split[ind + 1].split(',')[0])

count_sw_switch = collections.Counter(switch_lang)
print count_sw_switch
# count_en_switch = collections.Counter(switch_word_en)
# count_mix_switch = collections.Counter(switch_word_mix)
# count_other_switch = collections.Counter(switch_word_other)


# A_for_plot = []
# text_for_plot = []
# for item in count_en_switch.most_common(25):
# 	A_for_plot.append(item[1])
# for item in count_en_switch.most_common(25):
# 	text_for_plot.append(item[0])
# plt.figure(1)
# bar_width = 0.5
# opacity = 0.4
# index = np.arange(len(A_for_plot))
# # num_all.sort(reverse=True)
# rects1 = plt.bar(index+1, np.array(A_for_plot), bar_width,
#                  alpha=opacity,
#                  color='b',
#                  label='Interviewer'
#                  )

# plt.ylabel('Number of Occurrences')
# #plt.xticks(index + bar_width , tuple(range(len(num_inv1))))
# #plt.ylim( 0, 20 )
# plt.xticks([])
# for i,val in enumerate(A_for_plot):
#     plt.text(index[i] + 1,val + 120 * len(str(text_for_plot[i])) , str(text_for_plot[i]), color='black', fontsize = 20, fontweight='bold',rotation='vertical')
# plt.show()









