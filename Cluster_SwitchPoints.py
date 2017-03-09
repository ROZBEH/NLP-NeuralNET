'''In this code I am going to cluster different code switch points and see what are the clusters that they belong to'''

# -*- coding: utf-8 -*-
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

file_ = open(os.path.join(r"/Users/Rouzbeh/Google Drive/ESCALES/NLP/Switchpoint", "MetadataInterviews2.txt"), "r")
read_file = file_.read()
inview_split = read_file.split('|||')
# We want to see how does the ratio changes over time, we consider 1 ratio on every quarter
inv_switch_feat = []
inve_switch_feat = []
for inview in inview_split:
	this_inview = inview.split('\r\n')
	if this_inview[0] == "":
		for i in range(3,len(this_inview)):
			utter = this_inview[i].split('\t')
			utter_ = utter
			if utter == [""]:
				continue
			# print "---------------------"
			# print utter
			utter = [x for x in utter if x.split(',')[3]!="punc"]
			# print "++++++++++++++++++++"
			# print utter
			for ind,word in enumerate(utter):
				features = word.split(',')
				if i % 2 == 0:
					if features[-1] == '1':
						if ind-1 >= 0:
							feat0 = utter[ind-1].split(',')[3]
						else:
							feat0 = 'start'
						try:
							feat1 = utter[ind+1].split(',')[3]
						except IndexError:
							feat1 = 'end'
						inve_switch_feat.append([feat0,features[3],feat1])
				elif i % 2 == 1:
					if features[-1] == '1':
						if ind-1 >= 0:
							feat0 = utter[ind-1].split(',')[3]
						else:
							feat0 = 'start'
						try:
							feat1 = utter[ind+1].split(',')[3]
						except IndexError:
							feat1 = 'end'
						inv_switch_feat.append([feat0,features[3],feat1])
				
	else:
		for i in range(2,len(this_inview)):
			utter = this_inview[i].split('\t')
			utter_ = []
			if utter == [""]:
				continue
			# print "---------------------"
			# print utter
			utter = [x for x in utter if x.split(',')[3]!="punc"]
			# print "++++++++++++++++++++++++"
			# print utter
			for ind,word in enumerate(utter):
				features = word.split(',')
				if i % 2 == 0:
					if features[-1] == '1':
						if ind-1 >= 0:
							feat0 = utter[ind-1].split(',')[3]
						else:
							feat0 = 'start'
						try:
							feat1 = utter[ind+1].split(',')[3]
						except IndexError:
							feat1 = 'end'
						inv_switch_feat.append([feat0,features[3],feat1])
				elif i % 2 == 1:
					if features[-1] == '1':
						if ind-1 >= 0:
							feat0 = utter[ind-1].split(',')[3]
						else:
							feat0 = 'start'
						try:
							feat1 = utter[ind+1].split(',')[3]
						except IndexError:
							feat1 = 'end'
						inve_switch_feat.append([feat0,features[3],feat1])

file_.close()	


# print "Interviewer Switch", inv_switch_feat
# print "Interviewee Switch", inve_switch_feat
'''
creating a dictionary of all of the langs with their index for using in clustering
dict_lang = {'mixed': 3, 'start': 2, 'en': 0, 'sw': 1, 'other': 4}
'''
dict_lang ={}
index = 0
for item in inve_switch_feat:
	for sub in item:
		if sub not in dict_lang:
			dict_lang[sub] = index
			index += 1
for item in inv_switch_feat:
	for sub in item:
		if sub not in dict_lang:
			dict_lang[sub] = index
			index += 1

# Now lets cluster the labels and see what is different clusters in our list
unique_inve = [list(x) for x in set(tuple(x) for x in inve_switch_feat)]
unique_inv = [list(x) for x in set(tuple(x) for x in inv_switch_feat)]

count_inv = []
num_inv = []
count_inve = []
num_inve = []
for item in unique_inve:
	a = inve_switch_feat.count(item)
	count_inv.append(item + [a])
	num_inv.append(a)
for item in unique_inv:
	a = inv_switch_feat.count(item)
	count_inve.append(item + [a])
	num_inve.append(a)
print count_inve
print "\n"
print count_inv
print "\n"
num_inv1 = []
num_inv2 = []
for item in num_inv:
	if item > 40:
		num_inv2.append(item)
	else:
		num_inv1.append(item)

num_inve1 = []
num_inve2 = []
for item in num_inve:
	if item > 40:
		num_inve2.append(item)
	else:
		num_inve1.append(item)
'''
plt.figure(1)
bar_width = 0.8
opacity = 0.4
index = np.arange(len(num_inve1))
num_inve1.sort(reverse=True)
rects1 = plt.bar(index+1, np.array(num_inve1), bar_width,
                 alpha=opacity,
                 color='b',
                 #yerr=std_men,
                 #error_kw=error_config,
                 label='Interviewer')

plt.ylabel('Number of Occurrences')
#plt.xticks(index + bar_width , tuple(range(len(num_inv1))))
plt.ylim( 0, 25 )
plt.xticks([])
plt.show()
'''
swicth_all = inv_switch_feat + inve_switch_feat
unique_all = [list(x) for x in set(tuple(x) for x in swicth_all)]
num_all = []
count_all = []
for item in unique_all:
	a = swicth_all.count(item)
	count_all.append(item + [a])
	num_all.append(a)
num_all1 = []
num_all2 = []
for item in num_all:
	if item > 40:
		num_all2.append(item)
	else:
		num_all1.append(item)

print count_all
plt.figure(1)
bar_width = 0.8
opacity = 0.4
index = np.arange(len(num_all))
num_all.sort(reverse=True)
rects1 = plt.bar(index+1, np.array(num_all), bar_width,
                 alpha=opacity,
                 color='b',
                 #yerr=std_men,
                 #error_kw=error_config,
                 label='Interviewer')

plt.ylabel('Number of Occurrences')
#plt.xticks(index + bar_width , tuple(range(len(num_inv1))))
plt.ylim( 0, 5000 )
plt.xticks([])
plt.show()

