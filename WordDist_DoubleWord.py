'''In this code I am going to plot the distribution of the words befor and after the switch points'''
#@author rouzbeh
# -*- coding: utf-8 -*-
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import collections

file_ = open(os.path.join(r"/Users/Rouzbeh/Google Drive/ESCALES/NLP/Switchpoint", "MetadataInterviews2.txt"), "r")
read_file = file_.read()
inview_split = read_file.split('|||')
# We want to see how does the ratio changes over time, we consider 1 ratio on every quarter
inv_switch_en = []
inv_switch_sw = []
inv_switch_NN = []
inve_switch_en = []
inve_switch_sw = []
inve_switch_NN = []
for inview in inview_split:
	this_inview = inview.split('\r\n')
	if this_inview[0] == "":
		for i in range(3,len(this_inview)):
			utter = this_inview[i].split('\t')
			for ind,word in enumerate(utter):
				features = word.split(',')
				if i % 2 == 0:
					if features[-1] == '1':
						try:
							if utter[ind+1].split(',')[-1] != '1' and utter[ind+2].split(',')[-1] == '1':
								if utter[ind+1].split(',')[3] == "en":
									inve_switch_en.append(utter[ind+1].split(',')[0].lower() + " " + utter[ind+2].split(',')[0].lower())
								elif utter[ind+1].split(',')[3] == "sw":
									inve_switch_sw.append(utter[ind+1].split(',')[0].lower() + " " + utter[ind+2].split(',')[0].lower())
								elif utter[ind+1].split(',')[3] != "punc" and utter[ind+1].split(',')[3] != "sw" and utter[ind+1].split(',')[3] != "en":
									inve_switch_NN.append(utter[ind+1].split(',')[0].lower() + " " + utter[ind+2].split(',')[0].lower())
						except IndexError:
							'hehe'
						
				elif i % 2 == 1:
					if features[-1] == '1':
						try:
							if utter[ind+1].split(',')[-1] != '1' and utter[ind+2].split(',')[-1] == '1':
								if utter[ind+1].split(',')[3] == "en":
									inv_switch_en.append(utter[ind+1].split(',')[0].lower() + " " + utter[ind+2].split(',')[0].lower())
								elif utter[ind+1].split(',')[3] == "sw":
									inv_switch_sw.append(utter[ind+1].split(',')[0].lower() + " " + utter[ind+2].split(',')[0].lower())
								elif utter[ind+1].split(',')[3] != "punc" and utter[ind+1].split(',')[3] != "sw" and utter[ind+1].split(',')[3] != "en":
									inv_switch_NN.append(utter[ind+1].split(',')[0].lower() + " " + utter[ind+2].split(',')[0].lower())
						except IndexError:
							'hehe'
				
	else:
		for i in range(2,len(this_inview)):
			utter = this_inview[i].split('\t')
			for ind,word in enumerate(utter):
				features = word.split(',')
				if i % 2 == 0:
					if features[-1] == '1':
						try:
							if utter[ind+1].split(',')[-1] != '1' and utter[ind+2].split(',')[-1] == '1':
								if utter[ind+1].split(',')[3] == "en":
									inv_switch_en.append(utter[ind+1].split(',')[0].lower() + " " + utter[ind+2].split(',')[0].lower())
								elif utter[ind+1].split(',')[3] == "sw":
									inv_switch_sw.append(utter[ind+1].split(',')[0].lower() + " " + utter[ind+2].split(',')[0].lower())
								elif utter[ind+1].split(',')[3] != "punc" and utter[ind+1].split(',')[3] != "sw" and utter[ind+1].split(',')[3] != "en":
									inv_switch_NN.append(utter[ind+1].split(',')[0].lower() + " " + utter[ind+2].split(',')[0].lower())
						except IndexError:
							'hehe'
				elif i % 2 == 1:
					if features[-1] == '1':
						try:
							if utter[ind+1].split(',')[-1] != '1' and utter[ind+2].split(',')[-1] == '1':
								if utter[ind+1].split(',')[3] == "en":
									inve_switch_en.append(utter[ind+1].split(',')[0].lower() + " " + utter[ind+2].split(',')[0].lower())
								elif utter[ind+1].split(',')[3] == "sw":
									inve_switch_sw.append(utter[ind+1].split(',')[0].lower() + " " + utter[ind+2].split(',')[0].lower())
								elif utter[ind+1].split(',')[3] != "punc" and utter[ind+1].split(',')[3] != "sw" and utter[ind+1].split(',')[3] != "en":
									inve_switch_NN.append(utter[ind+1].split(',')[0].lower() + " " + utter[ind+2].split(',')[0].lower())
						except IndexError:
							'hehe'

file_.close()	

counter_en_inv = collections.Counter(inv_switch_en)
counter_sw_inv = collections.Counter(inv_switch_sw)
counter_NN_inv = collections.Counter(inv_switch_NN)
counter_en_inve = collections.Counter(inve_switch_en)
counter_sw_inve = collections.Counter(inve_switch_sw)
counter_NN_inve = collections.Counter(inve_switch_NN)
counter_sw_all = collections.Counter(inve_switch_sw + inv_switch_sw)
counter_en_all = collections.Counter(inve_switch_en + inv_switch_en)
counter_NN_all = collections.Counter(inve_switch_NN + inv_switch_NN)
# print(counter_en_inv.most_common(35))
# print(counter_sw_inv.most_common(35))
# print(counter_en_inve.most_common(35))
# print(counter_sw_inve.most_common(35))

A_for_plot = []
text_for_plot = []
for item in counter_sw_all.most_common(35):
	A_for_plot.append(item[1])
for item in counter_sw_all.most_common(35):
	text_for_plot.append(item[0])
plt.figure(1)
bar_width = 0.5
opacity = 0.4
index = np.arange(len(A_for_plot))
# num_all.sort(reverse=True)
rects1 = plt.bar(index+1, np.array(A_for_plot), bar_width,
                 alpha=opacity,
                 color='b',
                 label='Interviewer'
                 )

plt.ylabel('Number of Occurrences')
#plt.xticks(index + bar_width , tuple(range(len(num_inv1))))
plt.ylim( 0, 20 )
plt.xticks([])
for i,val in enumerate(A_for_plot):
    plt.text(index[i] + 1 ,val + len(str(text_for_plot[i]))/3 , str(text_for_plot[i]), color='blue', fontweight='bold',rotation='vertical')
plt.show()

















