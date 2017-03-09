'''In this code I am trying to run some simple statistics of our interview data and get some helpful data
about the interview'''

# -*- coding: utf-8 -*-
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

file_ = open(os.path.join(r"/Users/Rouzbeh/Google Drive/ESCALES/NLP/Switchpoint", "MetadataInterviews.txt"), "r")
read_file = file_.read()
inview_split = read_file.split('|||')
Iv_Sw = []
Ivee_Sw =[]
Iv_Sw_Ratio = []
Ivee_Sw_Ratio =[]
Inv_Wrd = []
Invee_Wrd = []
# We want to see how does the ratio changes over time, we consider 1 ratio on every quarter
ratio_inv_time = []
ratio_invee_time = []
name_inv = []
name_invee = []
gender_inv = []
gender_invee = []
for inview in inview_split:
	this_inview = inview.split('\r\n')
	sw_inv = 0
	sw_invee = 0
	wrd_inv = 0
	wrd_invee = 0
	ratio_iv = []
	ratio_ivee =[]
	if this_inview[0] == "":
		name_inv.append(this_inview[1].split(',')[1])
		gender_inv.append(this_inview[1].split(',')[4])
		name_invee.append(this_inview[2].split(',')[1])
		gender_invee.append(this_inview[2].split(',')[4])
		for i in range(3,len(this_inview)):
			utter = this_inview[i].split('\t')
			for word in utter:
				featurs = word.split(',')
				if i % 2 == 0:
					wrd_invee += 1
					if featurs[-1] == '1':
						sw_invee += 1
				elif i % 2 == 1:
					wrd_inv += 1
					if featurs[-1] == '1':
						sw_inv += 1
			if i % (len(this_inview)/70) == 0 :
				ratio_iv.append((sw_inv+1)/float(1+wrd_inv))
				ratio_ivee.append((sw_invee+1)/float(1+wrd_invee))
	else:
		name_inv.append(this_inview[0].split(',')[1])
		gender_inv.append(this_inview[0].split(',')[4])
		name_invee.append(this_inview[1].split(',')[1])
		gender_invee.append(this_inview[1].split(',')[4])
		for i in range(2,len(this_inview)):
			utter = this_inview[i].split('\t')
			for word in utter:
				featurs = word.split(',')
				if i % 2 == 0:
					wrd_inv += 1
					if featurs[-1] == '1':
						sw_inv += 1
				elif i % 2 == 1:
					wrd_invee += 1
					if featurs[-1] == '1':
						sw_invee += 1
			if i % (len(this_inview)/70) == 0 :
				ratio_iv.append((sw_inv+1)/float(1+wrd_inv))
				ratio_ivee.append((sw_invee+1)/float(1+wrd_invee))
	Iv_Sw.append(sw_inv)
	Ivee_Sw.append(sw_invee)
	# I multiply some of them by 100000 so that some of the elements does not vanish
	Iv_Sw_Ratio.append((sw_inv+1)/float(1+wrd_inv))
	Ivee_Sw_Ratio.append((1+sw_invee)/float(1+wrd_invee))
	Inv_Wrd.append(wrd_inv)
	Invee_Wrd.append(wrd_invee)
	ratio_inv_time.append(ratio_iv)
	ratio_invee_time.append(ratio_ivee)

print Iv_Sw
print Ivee_Sw
print "Interviewer Total Switch = ",sum(Iv_Sw)
print "Interviewee Total Switch = ",sum(Ivee_Sw)
print "Interviewer Switch Ratio = ", Iv_Sw_Ratio
print "Interviewee Switch Ratio = ", Ivee_Sw_Ratio
print "Total Switch = " , sum(Ivee_Sw) + sum(Iv_Sw)
# for i in range(len(Iv_Sw)):
# 	print "----------------------------------------------"
# 	print name_inv[i], '...',Iv_Sw_Ratio[i], '...',Ivee_Sw_Ratio[i],'...',name_invee[i]
# 	print name_inv[i], '...',Iv_Sw[i], '......',Ivee_Sw[i],'...',name_invee[i]
file_.close()	






plt.figure(1)
bar_width = 0.35
opacity = 0.4
index = np.arange(30)

rects1 = plt.bar(index, np.array(Iv_Sw_Ratio), bar_width,
                 alpha=opacity,
                 color='b',
                 #yerr=std_men,
                 #error_kw=error_config,
                 label='Interviewer')

rects2 = plt.bar(index + bar_width, np.array(Ivee_Sw_Ratio), bar_width,
                 alpha=opacity,
                 color='r',
                 #yerr=std_women,
                 #error_kw=error_config,
                 label='Interviewee')
plt.ylabel('Ration of Switch to total number of words used')
plt.xlabel('Participant and interviewer')
plt.xticks(index + bar_width , tuple(range(1,31)))
plt.plot(index + bar_width/2, np.array(Iv_Sw_Ratio), color='blue')
plt.plot(index + bar_width + bar_width/2, np.array(Ivee_Sw_Ratio), color='red')
plt.legend()
plt.show()


plt.figure(2)
for i in range(len(ratio_inv_time)):
	plt.subplot(3,10,i+1)

	x = np.arange(len(ratio_inv_time[i]))
	plt.plot(x, np.array(ratio_inv_time[i]), color='blue', label='Interviewer')
	plt.plot(x, np.array(ratio_invee_time[i]), color='red', label='Interviewee')
	plt.title(name_inv[i] + " " + gender_inv[i] + '\n' + name_invee[i] + " " + gender_invee[i],fontsize=10)
	plt.axis('off')
	plt.autoscale(enable=True)
plt.show()