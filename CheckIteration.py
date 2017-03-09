'''This script checks the interview files whether the utterances are one after each other
in other words. Interviewer, participant, interviewer,participant,...'''
# @author: Rouzbeh Shirvani twitter: @rouzbehshirvani
# -*- coding: utf-8 -*-
import os
import glob
from xlrd import open_workbook
#x_file = open(os.path.join(r"/Users/Rouzbeh/Google Drive/ESCALES/Data Sets/ESCALES African Interviews/Kenya - March 2016", "MetadataInterviews.txt"), "r")
all_files = []
path_file = r"/Users/Rouzbeh/Google Drive/ESCALES/Data Sets/ESCALES African Interviews/Kenya - March 2016/"
paths = glob.glob(path_file + "*2016.xlsx")
for file in paths:
	wb = open_workbook(file)
	for sheet in wb.sheets():
		num_row = sheet.nrows
		num_col = sheet.ncols
		for row in range(1,num_row/2):
			value = sheet.cell(row*2-1,2).value
			if value != "Interviewer":
				# Check if the specific cell of the excell file is "Interviewer", if not we might need to check
				# one cell after another should be interviewer and participant.
				print row*2-1
				print file