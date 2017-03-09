# -*- coding: utf-8 -*-
'''
This script tries to generate and put the features of the jamii forum data together. At the end we will have each word with the correspoding feature vector. The end file is stored inside "Feature_file.txt"

@authors: Rouzbeh and Mario
'''
from __future__ import unicode_literals
import re, os, glob, math, sys, time
from openpyxl.reader.excel import load_workbook
import treetaggerwrapper
from collections import defaultdict as dd
import csv
from _sqlite3 import Row
startTime = time.time()
csv_file = 'Context_Dependent_Predictions_PostNum_edit.csv'
tagdir = '/Users/Rouzbeh/BoxSync/Fall2015/ESCALES/SwahiliTagger'
path = '/Users/Rouzbeh/Google Drive/Language ID/'
'''
In this code I am generating feature vector for each word and save it in Featuer_file.txt.
then naive_method.py will use it then naive_method_readydata.py
'''
result = open('All_text_.txt','w')
testresult = open('Feature_file_.txt','w')
f = open('Logfile_.txt','w')
garbage = open('garbage_.txt','w')

def row_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

if __name__ == '__main__':
    
    dataDir = '/Users/Rouzbeh/Google Drive/Language ID'
    writeDir = '/Users/Rouzbeh/Google Drive/Language ID'
    row_num = row_len(csv_file)
    with open(csv_file, 'rU') as csvfile:
        mycsv = csv.reader(csvfile)

    
        enTagger = treetaggerwrapper.TreeTagger(TAGLANG='en',TAGDIR=tagdir,TAGOPT='-token -lemma -sgml -quiet -proto -hyphen-heuristics')
        swTagger = treetaggerwrapper.TreeTagger(TAGLANG='sw',TAGDIR=tagdir,TAGOPT='-token -lemma -sgml -quiet -proto -hyphen-heuristics')
    
        Num_Conv = 0
        flag = -1
        # Skip the header line
        #mycsv.next()
        string_current  = ''
        
        def string_append(string_current,row):
            try:
                if row[2] != 'punc':
                    if row[3] == 'initcap' or row[3] == 'capital':
                        string_current += row[0][1:-1].capitalize()
                        string_current += ' '
                    elif row[3] == 'allcaps':
                        string_current += row[0][1:-1].upper()
                        string_current += ' '
                    else:
                        string_current += row[0][1:-1]
                        string_current += ' '
                        
                elif row[2] == 'punc':
                    string_current += row[0][1]
                    string_current += ' '
            except:
                if len(row[0]) > 3:
                    #print row[0],
                    row0 = row[0]
                    
                if re.search(r"[^\w!\.?@#$%&'\-\\/+=]", row[0]):
                    row[0] = re.sub(r"[^\w!\.?@#$%&'\-\\/+=]","", row[0])
                    if row[0] == "##":
                        row[0] = "BAD_CHAR"
                    else:
                        #print "\t",row[0]
                        print >>garbage, row0, "---",row[0],"\n"
                else:
                    row[0] = "BAD_CHAR"
                if row[2] != 'punc':
                    if row[3] == 'initcap' or row[3] == 'capital':
                        string_current += row[0][1:-1].capitalize()
                        string_current += ' '
                    elif row[3] == 'allcaps':
                        string_current += row[0][1:-1].upper()
                        string_current += ' '
                    else:
                        string_current += row[0][1:-1]
                        string_current += ' '
                        
                elif row[2] == 'punc':
                    string_current += row[0][1]
                    string_current += ' '
            return string_current
        #featlabels = ("word","POS","lemma","lang","lang-1","lang-2","rlang-1","rlang-2","sameLang","diffLang","sameLangLog","diffLangLog","sameLangRatio","preCoSw","CoSwPoint")
        lang_list_all = []
        total_switch = 0
        All_string = []
        cct = 0
        progress = 0
        disp_prog = 10
        for row in mycsv:
            progress +=1
            if progress % (row_num/10) == 0:
                print disp_prog + " percent of the file has been processed."
                disp_prog += 10
            #print type(row[0])
            #print "hehe"
            if bool(re.search('Post:.', row[0])) == True:
                # This is the previous code switch flag
                cs_p = 0
                # This is the current code switch flag
                cs_c = 0
                en_lang = 0
                sw_lang = 0
                mix_lang = 0
                other_lang = 0
                #print lang_list_all
                langtag = ','.join(str(ee) for ee in lang_list_all)
                testresult.write(langtag)
                if string_current != '':
                    result.write(string_current)
                    result.write('\n')
                    All_string.append(string_current)
                string_current  = ''
                
                # In this section we are going to use the tagger to tag each chunck of the text
                cursor = 0
                swTags2 = []
                for sent in All_string:
                    #print cursor
                    #print lang_list_all
                    LangId = lang_list_all[cursor]
                    if LangId == 0:
                        Tags = swTagger.tag_text(unicode(sent))
                        for item in Tags:
                            swTags2.append(item.split()[0:3])
                    elif LangId == 1:
                        Tags = enTagger.tag_text(unicode(sent))
                        for item in Tags:
                            swTags2.append(item.split()[0:3])
                    elif LangId == 2:
                        Tags = swTagger.tag_text(unicode(sent))
                        for item in Tags:
                            swTags2.append(item.split()[0:3])
                    elif LangId == 3:
                        # for the other words the tag is unknown
                        sent1 = sent.split(' ')
                        for ii in range(len(sent1)):
                            temp1 = [sent1[ii], '<unknown>', sent1[ii]]
                            swTags2.append(temp1)
                        Tags = swTags2
                    elif LangId == 4:
                        sent1 = sent.split(' ')
                        for ii in range(len(sent1)):
                            temp1 = [sent1[ii], 'SENT', sent1[ii]]
                            swTags2.append(temp1)
                        Tags = swTags2
                    
                    add_diff = len(Tags) - len(sent.split())
                    for i in range(add_diff):
                        lang_list_all.insert(cursor,LangId)
                    cursor += len(Tags)


                # In this section we are going to build the feature vector for each non punctuation words.
                lang_list_final = lang_list_all
                #Remove the punctuations from lang_list_all
                lang_list_all = [x for x in lang_list_all if x != 4]
                vec_feat =[[]]*len(lang_list_all)
                for aa in range(len(lang_list_all)):
                    templist = []
                    if lang_list_all[aa] == 0 :
                        # 0 for swahili, 1 for english, 2 for mixed, 3 for other, -1 for init, 4 for pun
                        templist.append(0)
                    elif lang_list_all[aa] == 1 :
                        templist.append(1)
                    elif lang_list_all[aa] == 2 :
                        templist.append(2)
                    elif lang_list_all[aa] == 3 :
                        templist.append(3)
                    if aa-1 >= 0:
                        if lang_list_all[aa-1] == 0 :
                            templist.append(0)
                        elif lang_list_all[aa-1] == 1 :
                            templist.append(1)
                        elif lang_list_all[aa-1] == 2 :
                            templist.append(2)
                        elif lang_list_all[aa-1] == 3 :
                            templist.append(3)
                    else:
                        templist.append(-1)
                    if aa-2 >= 0:
                        if lang_list_all[aa-2] == 0 :
                            templist.append(0)
                        elif lang_list_all[aa-2] == 1 :
                            templist.append(1)
                        elif lang_list_all[aa-2] == 2 :
                            templist.append(2)
                        elif lang_list_all[aa-1] == 3 :
                            templist.append(3)
                    else:
                        templist.append(-1)
                    if aa-1 >= 0:
                        if lang_list_all[aa] == lang_list_all[aa-1] :
                            # same labeled as 0, diff labeled as 1
                            templist.append(0)
                            if lang_list_all[aa] == 0:
                                sw_lang += 1
                            elif lang_list_all[aa] == 1:
                                en_lang += 1
                            elif lang_list_all[aa] == 2:
                                mix_lang += 1
                            elif lang_list_all[aa] == 3:
                                other_lang += 1
                        else:
                            cs_p = 1
                            if lang_list_all[aa] == 2:
                                templist.append(1)
                                mix_lang += 1    
                            elif lang_list_all[aa] == 1:
                                templist.append(1)
                                en_lang += 1
                            elif lang_list_all[aa] == 0:
                                templist.append(1)
                                sw_lang += 1
                            elif lang_list_all[aa] == 3:
                                templist.append(1)
                                other_lang += 1
                    else:
                        if lang_list_all[aa] == 0:
                            sw_lang += 1
                        elif lang_list_all[aa] == 1:
                            en_lang += 1
                        elif lang_list_all[aa] == 2:
                            mix_lang += 1
                        elif lang_list_all[aa] == 3:
                            other_lang += 1
                        templist.append(-1)
                    if aa-2 >= 0:
                        if lang_list_all[aa] == lang_list_all[aa-2] :
                            templist.append(0)
                        else:
                            templist.append(1)
                    else:
                        templist.append(-1)
                    if lang_list_all[aa] == 0:
                        templist.append(sw_lang)
                        templist.append(en_lang+mix_lang+other_lang)
                        templist.append(round(math.log((sw_lang+1),2),3))
                        templist.append(round(math.log((en_lang+mix_lang+other_lang+1),2),3))
                        templist.append(round(sw_lang/float(sw_lang+en_lang+mix_lang+other_lang),3))
                    elif lang_list_all[aa] == 1:
                        templist.append(en_lang)
                        templist.append(sw_lang+mix_lang+other_lang)
                        templist.append(round(math.log((en_lang+1),2),3))
                        templist.append(round(math.log((sw_lang+mix_lang+other_lang+1),2),3))
                        templist.append(round(en_lang/float(sw_lang+en_lang+mix_lang+other_lang),3))
                    elif lang_list_all[aa] == 2:
                        templist.append(mix_lang)
                        templist.append(en_lang+sw_lang+other_lang) 
                        templist.append(round(math.log((mix_lang+1),2),3))
                        templist.append(round(math.log((en_lang+sw_lang+other_lang+1),2),3))
                        templist.append(round(mix_lang/float(sw_lang+en_lang+mix_lang+other_lang),3))
                    elif lang_list_all[aa] == 3:
                        templist.append(other_lang)
                        templist.append(en_lang+sw_lang+mix_lang) 
                        templist.append(round(math.log((other_lang+1),2),3))
                        templist.append(round(math.log((en_lang+sw_lang+mix_lang+1),2),3))
                        templist.append(round(mix_lang/float(sw_lang+en_lang+mix_lang+other_lang),3))

                    templist.append(cs_p)
                    if aa+1 < len(lang_list_all):
                        if lang_list_all[aa] != lang_list_all[aa+1]:
                            cs_c = 1
                            total_switch += 1
                    templist.append(cs_c)
                    cs_c = 0
                    vec_feat[aa] = templist  
                for al in vec_feat:
                    al = ','.join(str(e) for e in al)
                    testresult.write("\t")
                    testresult.write(al)
                    
                    
                    
                    
                ## Appending all features together for the final feature vector   
                punc_flag = -1    
                punc_cursor = 0
                Final_Featt_All =[]
                for indix in range(len(lang_list_final)):
                    Final_Feat = [] 
                    if lang_list_final[indix] != 4:
                        Final_Feat.append(lang_list_final[indix])
                        punc_flag = lang_list_final[indix]
                        try:
                            tempor =[value.encode('ascii','ignore') for value in swTags2[indix]]
                        except:
                            print "index:",indix
                            print len(swTags2),":",swTags2
                            print len(lang_list_final),":",lang_list_final
                            print len(lang_list_all),":",lang_list_all
                            print "total_switch = ",total_switch
                            print "Total words = ",cct
                            os.system('say "I need you to fix me"')
                            garbage.close()
                            
#                             print swTags2[indix]
                            sys.exit()
                        Final_Feat = Final_Feat + vec_feat[indix-punc_cursor] + tempor
                    elif lang_list_final[indix] == 4:
                        punc_cursor += 1
                        if punc_flag == 0:
                            # Flag for this kind of punctuation
                            Final_Feat.append(4)
                        elif punc_flag == 1:
                            # Flag for this kind of punctuation
                            Final_Feat.append(5)
                        elif punc_flag == 2: 
                            # Flag for this kind of punctuation   
                            Final_Feat.append(6)
                        elif punc_flag == 3:
                            Final_Feat.append(7)
    
                        tempor =[value.encode('ascii','ignore') for value in swTags2[indix]]
                        Final_Feat = Final_Feat + vec_feat[indix-punc_cursor] + tempor
                    # This feature is repeated two times, I am just removing it.
                    del Final_Feat[1]
#                     if not Final_Feat:
                    print >>f, Final_Feat
                print >>f, '\n'
                print >>f, All_string
                print >>f, '\n'
                    
                    
                    
                    
                testresult.write('\n') 
                #lang_list_all list of all language tags including punctuations, swahili, English, others and mixed
                lang_list_all = []
                All_string = []
                Num_Conv += 1
                flag = 0
            elif bool(re.search('Post:.', row[0])) == False:
                cct += 1
                if flag == 0:
                    string_current = string_append(string_current,row)
                    if row[2] == 'en':
                        lang_list_all.append(1)
                    elif row[2] == 'sw':
                        lang_list_all.append(0)
                    elif row[2] == 'mixed':
                        lang_list_all.append(2)
                    elif row[2] == 'other':
                        lang_list_all.append(3)
                    elif row[2] == 'punc':
                        lang_list_all.append(4)
                        
                    flag = 1
                    temp = row[2]
                elif flag == 1:
                    if temp != row[2]:
                        if row[2] == 'en' :
                            #enTags = enTagger.tag_text(unicode(string_current))
                            All_string.append(string_current)
                            result.write('\n')
                            result.write(string_current)
                            string_current = ''
                            string_current = string_append(string_current,row)
                            if row[2] == 'en':
                                lang_list_all.append(1)
                            elif row[2] == 'sw':
                                lang_list_all.append(0)
                            elif row[2] == 'mixed':
                                lang_list_all.append(2)
                            elif row[2] == 'other':
                                lang_list_all.append(3)
                            elif row[2] == 'punc':
                                lang_list_all.append(4)
                            temp = row[2]
                        elif row[2] == 'sw':
                            result.write(string_current)
                            result.write('\n')
                            All_string.append(string_current)
                            string_current = ''
                            #swTags = swTagger.tag_text(unicode(string_current))
                            string_current = ''
                            string_current = string_append(string_current,row)
                            if row[2] == 'en':
                                lang_list_all.append(1)
                            elif row[2] == 'sw':
                                lang_list_all.append(0)
                            elif row[2] == 'mixed':
                                lang_list_all.append(2)
                            elif row[2] == 'other':
                                lang_list_all.append(3)
                            elif row[2] == 'punc':
                                lang_list_all.append(4)
                            temp = row[2]
                        elif row[2] == 'mixed':
                            result.write(string_current)
                            result.write('\n')
                            All_string.append(string_current)
                            string_current = ''
                            string_current = string_append(string_current,row)
                            if row[2] == 'en':
                                lang_list_all.append(1)
                            elif row[2] == 'sw':
                                lang_list_all.append(0)
                            elif row[2] == 'mixed':
                                lang_list_all.append(2)
                            elif row[2] == 'other':
                                lang_list_all.append(3)
                            elif row[2] == 'punc':
                                lang_list_all.append(4)
                            temp = row[2]
                            #We will have unknown Tag for mixed language word
                        elif row[2] == 'punc':
                            string_current = string_append(string_current,row)
                            if row[2] == 'en':
                                lang_list_all.append(1)
                            elif row[2] == 'sw':
                                lang_list_all.append(0)
                            elif row[2] == 'mixed':
                                lang_list_all.append(2)
                            elif row[2] == 'other':
                                lang_list_all.append(3)
                            elif row[2] == 'punc':
                                lang_list_all.append(4)
                    else:
                        string_current = string_append(string_current,row)
                        if row[2] == 'en':
                            lang_list_all.append(1)
                        elif row[2] == 'sw':
                            lang_list_all.append(0)
                        elif row[2] == 'mixed':
                            lang_list_all.append(2)
                        elif row[2] == 'other':
                            lang_list_all.append(3)
                        elif row[2] == 'punc':
                            lang_list_all.append(4)
                        temp = row[2]
                  
        
            
            
            
            
        result.write(string_current)
print "total_switch = ",total_switch
print "Total words = ",cct     
print "Finished ! :-)"
result.close()
testresult.close()
f.close()
garbage.close()
print ('The script took {0} second !'.format(time.time() - startTime))
os.system('say "your program has finished"')

    # Mixed is also considered as a switch
    ### Just ignore the punctuation and work with either Swahili or English or POST number to differentiate between different chunks 
    
