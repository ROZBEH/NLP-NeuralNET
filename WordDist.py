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
inve_switch_en = []
inve_switch_sw = []
for inview in inview_split:
	this_inview = inview.split('\r\n')
	if this_inview[0] == "":
		for i in range(3,len(this_inview)):
			utter = this_inview[i].split('\t')
			for ind,word in enumerate(utter):
				features = word.split(',')
				if i % 2 == 0:
					if features[-1] == '1':
						if features[3] == "en":
							inve_switch_en.append(features[0].lower())
						elif features[3] == "sw":
							inve_switch_sw.append(features[0].lower())
				elif i % 2 == 1:
					if features[-1] == '1':
						if features[3] == "en":
							inv_switch_en.append(features[0].lower())
						elif features[3] == "sw":
							inv_switch_sw.append(features[0].lower())
				
	else:
		for i in range(2,len(this_inview)):
			utter = this_inview[i].split('\t')
			for ind,word in enumerate(utter):
				features = word.split(',')
				if i % 2 == 0:
					if features[-1] == '1':
						if features[3] == "en":
							inv_switch_en.append(features[0].lower())
						elif features[3] == "sw":
							inv_switch_sw.append(features[0].lower())
				elif i % 2 == 1:
					if features[-1] == '1':
						if features[3] == "en":
							inve_switch_en.append(features[0].lower())
						elif features[3] == "sw":
							inve_switch_sw.append(features[0].lower())

file_.close()	

counter_en_inv = collections.Counter(inv_switch_en)
counter_sw_inv = collections.Counter(inv_switch_sw)
counter_en_inve = collections.Counter(inve_switch_en)
counter_sw_inve = collections.Counter(inve_switch_sw)
counter_sw_all = collections.Counter(inve_switch_sw + inv_switch_sw)
counter_en_all = collections.Counter(inve_switch_en + inv_switch_en)
# print(counter_en_inv.most_common(35))
# print(counter_sw_inv.most_common(35))
# print(counter_en_inve.most_common(35))
# print(counter_sw_inve.most_common(35))

A_for_plot = []
text_for_plot = []
for item in counter_en_inve.most_common(35):
	A_for_plot.append(item[1])
for item in counter_en_inve.most_common(35):
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
plt.ylim( 0, 150 )
plt.xticks([])
for i,val in enumerate(A_for_plot):
    plt.text(index[i] + 1 ,val + 7 + len(str(text_for_plot[i])) , str(text_for_plot[i]), color='blue', fontweight='bold',rotation='vertical')
plt.show()

















