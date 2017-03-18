#! /usr/bin/env python
import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import re
import os
import time
from datetime import datetime
import pickle
from gensim.models import word2vec
from sklearn.preprocessing import OneHotEncoder
import warnings
import RNNM2MClass as M2M
warnings.filterwarnings("ignore")
enc = OneHotEncoder()

_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '100'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.00625'))
_NEPOCH = int(os.environ.get('NEPOCH', '30'))
evaluate_loss_after = 2
# I just want to have all of the word tokens that's why I used 10823 here
vocabulary_size = 3000

_MODEL_FILE = os.environ.get('MODEL_FILE')
#vector_size = _VECTOR_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
hidden_dim = _HIDDEN_DIM
learning_rate = _LEARNING_RATE
nepoch = _NEPOCH
print "Reading the input file"
inputfile = open(os.path.join(r"/Users/Rouzbeh/Google Drive/ESCALES/NLP/Switchpoint", "MetadataInterviews2.txt"), "r")

read_file = inputfile.read()
inview_split = read_file.split('|||')
# List of all words in the interview file
tokenized_sentences = []
switch_label = []
feat_vec = []
feat_1hot = []
# all language labels that are available such as 'en', 'sw', ...
dic0 = {'en':0, 'sw':1, 'other':2, 'punc':4, 'mixed':3}
dic1 = {'init':0, 'en':1, 'sw':2, 'other':3, 'punc':5, 'mixed':4}
dic2 = {'init': 0, 'same':1, 'diff':2 }
for inview in inview_split:
    this_inview = inview.split('\r\n')
    if this_inview[0] == "":
        for i in range(3,len(this_inview)):
            utter = this_inview[i].split('\t')
            sent = []
            switch = []
            all_feat = []
            # I am adding the previous output as the input to the next layer
            sw = 0
            for word in utter:
                feat = []
                try:
                    features = word.split(',')
                    if features[3] != 'punc':
                        feat.append(dic0[features[3]])
                        feat.append(dic1[features[4]])
                        feat.append(dic1[features[5]])
                        feat.append(dic2[features[6]])
                        feat.append(dic2[features[7]])
                        feat.append(sw)
                        sent.append(features[0])
                        switch.append(int(features[-1]))
                        sw = int(features[-1])
                        all_feat.append(feat)
                        feat_1hot.append(feat)
                except IndexError:
                    "hehe"
                
            tokenized_sentences.append(sent)
            switch_label.append(switch)
            feat_vec.append(all_feat)
                
    else:
        for i in range(2,len(this_inview)):
            utter = this_inview[i].split('\t')
            sent = []
            switch = []
            all_feat = []
            # I am adding the previous output as the input to the next layer
            sw = 0
            for word in utter:
                feat = []
                try:
                    features = word.split(',')
                    if features[3] != 'punc':
                        feat.append(dic0[features[3]])
                        feat.append(dic1[features[4]])
                        feat.append(dic1[features[5]])
                        feat.append(dic2[features[6]])
                        feat.append(dic2[features[7]])
                        feat.append(sw)
                        sent.append(features[0])
                        switch.append(int(features[-1]))
                        sw = int(features[-1])
                        all_feat.append(feat)
                        feat_1hot.append(feat)
                except IndexError:
                    "hehe"
                
            tokenized_sentences.append(sent)
            switch_label.append(switch)
            feat_vec.append(all_feat)
                       
inputfile.close()
# Here we are chaning the feature vector into a one hot representation
enc.fit(feat_1hot)
for elems in feat_vec:
    for indix, val in enumerate(elems):
        elems[indix] = enc.transform(val).toarray().T
# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())
a = 0
for item in tokenized_sentences:
    a += len(item)
print "Total number of non-punctuation words is = ", a
# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
# Create the training data
X = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_sentences])
y = np.asarray(switch_label)
# because I am adding some other features at the end of the input vector
# Second set of features have the size of 24 that are concatenated at the end of the input vector
features2 = 24
X_train = X[0:int(.8*X.shape[0])]
y_train = y[0:int(.8*y.shape[0])]
X_test = X[int(.8*X.shape[0]):]
y_test = y[int(.8*y.shape[0]):]
feat_vec_train = feat_vec[0:int(.8*len(feat_vec))]
feat_vec_test = feat_vec[int(.8*len(feat_vec)):]






# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, feat_vec_train, learning_rate=0.005, nepoch=10, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    print "There is going to be ", nepoch, " epoches"
    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train,feat_vec_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen = %d epoch = %d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # For each training example...
        print "Starting Epoch ", epoch
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], feat_vec_train[i], learning_rate)
            num_examples_seen += 1
    filename11 = 'numpy_model.sav'
    pickle.dump(model, open(filename11, 'wb'))

# this function is for the classification task of the output labels and the correspoding classification metrics
def predict_label(model, X_test,y_test,feat_vec_test):
    # Metrics here are self defining, one can understanf just by following them
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(y_test)):
        predicted = model.predict(X_test[i],feat_vec_test[i])
        #print predicted
        predicted = predicted.tolist()
        for j in range(len(y_test[i])):
            if predicted[j] == y_test[i][j]:
                if predicted[j] == 1:
                    true_pos += 1
                else:
                    true_neg += 1
            else:
                if predicted[j] == 1:
                    false_pos += 1
                else:
                    false_neg += 1
    print "true_pos",true_pos, "false_pos", false_pos
    precision = (true_pos)/float(true_pos + false_pos)
    recall = (true_pos)/float(true_pos + false_neg)
    fscore = (2*precision*recall)/float(precision+recall)
    accuracy = (true_neg + true_pos)/float(true_neg + true_pos + false_neg + false_pos)
    return precision, recall, fscore, accuracy


model = M2M.RNNNumpy(vocabulary_size + features2, hidden_dim)
train_with_sgd(model, X_train, y_train, feat_vec_train, learning_rate, nepoch, evaluate_loss_after)



# filename11 = 'numpy_model.sav'
# model = pickle.load(open(filename11, 'rb'))
# Calling predict_label function and printing the results
print " precision, recall, fscore, accuracy = " , predict_label(model, X_test,y_test,feat_vec_test), "Test Data"
print " precision, recall, fscore, accuracy = " , predict_label(model, X_train,y_train,feat_vec_train), "Train Data"
# tokenized_sentences
# At the end we print the constant that we used during this run of the simluation
print 'nepoch = ',nepoch ,'vocabulary_size = ' ,vocabulary_size ,'hidden_dim = ' ,hidden_dim ,'learning_rate = ',learning_rate
