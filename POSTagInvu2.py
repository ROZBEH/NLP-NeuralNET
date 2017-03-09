# -*- coding: utf-8 -*-
'''
Created on Nov 1, 2015

adds interview metadata at the beginning of each interview
adds features to the words in an utterance, but keeps 1 utterance per line

Output format:
interviewer data
participant data
interviewer: each,0,0,init,init\tword,0,0,en,init\tlike,0,0,en,en\tthis,0,0,en,en
participant: utterances
more utterances
|||
participant data
etc.

@author: Mario
'''

import re, os, glob, math,sys
from openpyxl.reader.excel import load_workbook
import treetaggerwrapper
from collections import defaultdict as dd

# import csv
# csv.field_size_limit(1000000000)

if __name__ == '__main__':
    
    dataDir = r"/Users/Rouzbeh/Google Drive/ESCALES/Data Sets/ESCALES African Interviews"
    
    metadataFile = open(dataDir + "/KenyaInterviewMetadata.csv")
    metadataFile.readline()
    
    metadata = {}
    for line in metadataFile:
      info = line.strip().split(",")
      transcript = info[0]
      role = info[1]
      metadata[(transcript,role)]=info[2:]
    
    dataDir = r"/Users/Rouzbeh/Google Drive/ESCALES/Data Sets/ESCALES African Interviews/Kenya - March 2016/"
    
    writeDir = r"/Users/Rouzbeh/Google Drive/ESCALES/NLP/Switchpoint"
    
    taggedFile = open(writeDir + "/MetadataInterviews.txt",'w')
    
    fileCount = 0
    utteranceCount = 0
    wordCount = 0
    for filename in glob.glob(os.path.join(dataDir, '*.xlsx')):
      if re.search("~\$",filename) or re.search("(unclean|split|version2)",filename):
        continue
#       if fileCount >= 1:
#         continue
      fileCount += 1
      print filename
      
      transcript = filename.split("\\")[-1].split()[0]
      if fileCount>1:
        taggedFile.write("|||\n")
      taggedFile.write("interviewer,"+",".join(metadata[(transcript,"interviewer")])+"\n")
      taggedFile.write("participant,"+",".join(metadata[(transcript,"participant")])+"\n")
      
      print "interviewer,"+",".join(metadata[(transcript,"interviewer")])
      print "participant,"+",".join(metadata[(transcript,"participant")])
      
      wb = load_workbook(filename)
    
      sheet = wb.active
      dialogue = []
      
      for cell in sheet['E']:
        if cell.value:
          try:
            dialogue.append(unicode(cell.value,"utf8"))
          except:
            dialogue.append(cell.value)
      
      for lineNum in range(1,len(dialogue)):
        utterance = dialogue[lineNum]
        utteranceCount += 1
        '''Remove hidden characters'''
        utterance = re.sub(u"\u001A","",utterance,0,re.UNICODE)
        '''Space out language tags and remove tag labels'''
        utterance = re.sub("<"," <",utterance,0,re.UNICODE)
        utterance = re.sub(">","> ",utterance,0,re.UNICODE)
        utterance = re.sub(" ?label=[\"'][\w ]+?[\"']","",utterance,0,re.I)
        utterance = re.sub(u"[\.\x85\u2026]+",".",utterance,0,re.UNICODE)
        utterance = re.sub(u"$[?!\.]+","",utterance,0,re.UNICODE)
        utterance = re.sub(u"(?<=[?!\.])"," ",utterance,0,re.UNICODE)
        '''Replace smart quotes and en/em-dashes with ASCII equivalents'''
        utterance = re.sub(u"[\u0027\x91\x92\u2018\u2019\u201b\u0060\u00b4\u02b9\u02bb\u02bc\u02bd\u02be\u02bf\u02ca\u02cb\u2018\u2019\u201b\u2032]+","'",utterance,0,re.UNICODE)
        utterance = re.sub("[\x96\x97]+","-",utterance,0,re.UNICODE)
        '''Remove laughter tokens and comments'''
        utterance = re.sub("\w+_token","",utterance,0,re.UNICODE)
        utterance = re.sub("[\(\[][\w ]+?[\)\]]","",utterance,0,re.UNICODE)
        '''Remove other punctuation''' #\u002f = /        
        utterance = re.sub(u"[=,;{}\(\)\[\]\x93\x94\xc2\xa0\"\u02ba\u02dd\u201c\u201d\u201e\u201f\u02ee\u2033\u2034\u00ab\u00bb\u00bf\u007c]+"," ",utterance,0,re.UNICODE)
        utterance = re.sub(u"(?<=[^\d])\:+(?=[^\d])","",utterance,0,re.UNICODE)
        utterance = re.sub("[\s\xa0]","  ",utterance,0,re.UNICODE)
        '''Remove tokens that don't contain letters or numbers'''
        utterance = re.sub("\b[^A-Za-z0-9]+\b","",utterance,0,re.UNICODE)
        utterance = re.sub("\s+"," ",utterance,0,re.UNICODE)
        '''Remove apostrophes, dashes and colons at beginnings/ends of words'''
        utterance = re.sub("( [\'\-]|[\'\-] )"," ",utterance,0,re.UNICODE)
        utterance = re.sub(u"$[\'\-]","",utterance,0,re.UNICODE)
        utterance = re.sub(u"[\'\-]$","",utterance,0,re.UNICODE)
        utterance = re.sub(u"-+","-",utterance,0,re.UNICODE)
        utterance = re.sub(u"\:+",":",utterance,0,re.UNICODE)
        utterance = re.sub(u"($\:|\:$)","",utterance,0,re.UNICODE)
        utterance = utterance.strip()
        if not re.search("\w",utterance):
          continue
#         utterance = utterance.lower()
        wordCount += len(utterance.split())
        
#         noTagUtter = re.sub("<[\\\\/]?\w+>","",utterance,0,re.UNICODE)
#         print "Full: ",utterance,"\n"
        
        enTagger = treetaggerwrapper.TreeTagger(TAGLANG='en',TAGOPT='-token -lemma -sgml -quiet -proto -hyphen-heuristics')
        swTagger = treetaggerwrapper.TreeTagger(TAGLANG='sw',TAGOPT='-token -lemma -sgml -quiet -proto -hyphen-heuristics')
        
        language = "en"
        chunk = u""
        utterTags = []
        featlabels = ("word","POS","lemma","lang","lang-1","lang-2","rlang-1","rlang-2","sameLang","diffLang","sameLangLog","diffLangLog","sameLangRatio","preCoSw","CoSwPoint")
        for word in utterance.split():
          if re.match("<mixed>",word,re.I):
            if language == "en" and len(chunk) > 0:
              enTags = enTagger.tag_text(chunk)
              for i in range(len(enTags)):
                enTags[i] = enTags[i].split()[0:3]
                enTags[i].append(language)
                utterTags.append(enTags[i])
            elif language == "sw":
              swTags = swTagger.tag_text(chunk)
              for i in range(len(swTags)):
                swTags[i] = swTags[i].split()[0:3]
                swTags[i].append(language)
                utterTags.append(swTags[i])
            language = "mixed"
            chunk = u""
          elif language == "mixed":
            if re.match("</mixed>",word,re.I):
              utterTags.append([chunk,"UNK","<unknown>","mixed"])
              language = "en"
              chunk = u""
            elif re.match("<.*>",word):
              continue
            else:
              chunk = chunk + word
          elif re.match("<(swahili|sheng)>",word,re.I):
            if language == "en" and len(chunk) > 0:
              enTags = enTagger.tag_text(chunk)
              for i in range(len(enTags)):
                enTags[i] = enTags[i].split()[0:3]
                enTags[i].append(language)
                utterTags.append(enTags[i])
            language = "sw"
            chunk = u""
          elif re.match("(<[\\\\/]\w+>|<english>)",word,re.I):
            if language == "sw":
              swTags = swTagger.tag_text(chunk)
              for i in range(len(swTags)):
                swTags[i] = swTags[i].split()[0:3]
                swTags[i].append(language)
                utterTags.append(swTags[i])
            language = "en"
            chunk = u""
          elif re.match("<\w+>",word):
            if language == "en" and len(chunk) > 0:
              enTags = enTagger.tag_text(chunk)
              for i in range(len(enTags)):
                enTags[i] = enTags[i].split()[0:3]
                enTags[i].append(language)
                utterTags.append(enTags[i])
            elif language == "sw":
              swTags = swTagger.tag_text(chunk)
              for i in range(len(swTags)):
                swTags[i] = swTags[i].split()[0:3]
                swTags[i].append(language)
                utterTags.append(swTags[i])
            language = "other"
            chunk = u""
          elif language != "other":
            chunk = chunk + " " + word
          else:
            word = re.sub("\.","",word)
            utterTags.append([word,"UNK","<unknown>","other"])
        
        if language == "en" and len(chunk) > 0:
          enTags = enTagger.tag_text(chunk)
          for i in range(len(enTags)):
            enTags[i] = enTags[i].split()[0:3]
            enTags[i].append(language)
            utterTags.append(enTags[i])
        
        prevswitch = "0"
        for i in range(len(utterTags)):
          if i > 0:
            if utterTags[i][3] != utterTags[i-1][3]:
              prevswitch = "1"
          utterTags[i].append(prevswitch)
        
        langcounts = dd(int)
        utterWC = 0.0
        for i in range(len(utterTags)):
          '''fix some punctuation tags'''
          if not re.search("\w",utterTags[i][0],re.UNICODE):
            utterTags[i][1] = "SENT"
            utterTags[i][3] = "punc"
            if i > 0:
              if utterTags[i][4] != utterTags[i-1][6]:
                utterTags[i][4] = utterTags[i-1][6]
          '''Add language of the previous token and the token before that'''
          if i > 1:
            utterTags[i].insert(-1,utterTags[i-1][3])
            utterTags[i].insert(-1,utterTags[i-2][3])
            if utterTags[i-1][3] == utterTags[i][3]:
              utterTags[i].insert(-1,"same")
            else:
              utterTags[i].insert(-1,"diff")
            if utterTags[i-2][3] == utterTags[i][3]:
              utterTags[i].insert(-1,"same")
            else:
              utterTags[i].insert(-1,"diff")
          elif i > 0:
            utterTags[i].insert(-1,utterTags[i-1][3])
            utterTags[i].insert(-1,"init")
            if utterTags[i-1][3] == utterTags[i][3]:
              utterTags[i].insert(-1,"same")
            else:
              utterTags[i].insert(-1,"diff")
            utterTags[i].insert(-1,"init")
          else:
            utterTags[i].insert(-1,"init")
            utterTags[i].insert(-1,"init")
            utterTags[i].insert(-1,"init")
            utterTags[i].insert(-1,"init")
          if utterTags[i][3] != "punc":
            langcounts[utterTags[i][3]] += 1
            utterWC += 1.0
            utterTags[i].insert(-1,langcounts[utterTags[i][3]])
            utterTags[i].insert(-1,int(utterWC)-langcounts[utterTags[i][3]])
            utterTags[i].insert(-1,math.log(langcounts[utterTags[i][3]]+1,2))
            utterTags[i].insert(-1,math.log(utterWC-langcounts[utterTags[i][3]]+1,2))
            utterTags[i].insert(-1,langcounts[utterTags[i][3]]/utterWC)
          else:
            utterTags[i].insert(-1,0)
            utterTags[i].insert(-1,0)
            utterTags[i].insert(-1,0)
        for i in range(len(utterTags)):  
          '''Check if the next word is in a different language, but ignore punctuation'''
          if i+1 < len(utterTags):
            if utterTags[i][3] == "mixed":
              utterTags[i].append("1")
            elif utterTags[i][3] == "punc":
              utterTags[i].append("0")
            #TODO: check if this altered code handles switches across punctuation correctly
            else:
              for j in range(i+1,len(utterTags)):
                if utterTags[j][3] == "punc":
                  if j+1 == len(utterTags):
                    utterTags[i].append("0")
                  else:
                    continue
                elif utterTags[i][3] != utterTags[j][3]:
                  utterTags[i].append("1")
                  break
                else:
                  utterTags[i].append("0")
                  break 
#             elif utterTags[i+1][3] == "punc":
#               if i+2 < len(utterTags):
#                 if utterTags[i][3] != utterTags[i+2][3]:
#                   utterTags[i].append("1")
#                 else:
#                   utterTags[i].append("0")
#               else:
#                 utterTags[i].append("0")
#             elif utterTags[i][3] != utterTags[i+1][3]:
#               utterTags[i].append("1")
#             else:
#               utterTags[i].append("0")
          else:
            utterTags[i].append("0")
        
#         for i in range(len(featlabels)):
#           print featlabels[i]+":\t",
#           for tag in utterTags:
#             print tag[i]+"\t",
#           print
#         print
        
        if len(utterTags) > 0:
          for i in range(len(utterTags)):
            for j in range(len(utterTags[i])):
              taggedFile.write(str(utterTags[i][j]))
              if j < len(utterTags[i])-1:
                taggedFile.write(",")
            if i < len(utterTags)-1:
              taggedFile.write("\t")
          taggedFile.write("\n")
        
#         if utteranceCount > 10:
#           break
        
        if utteranceCount%1000 == 0:
          print utteranceCount,"utterances done"
    
    print "\nAll",fileCount,"interviews done. Files closed."
    print utteranceCount,"utterances with",wordCount,"words\n"
       
    taggedFile.close()