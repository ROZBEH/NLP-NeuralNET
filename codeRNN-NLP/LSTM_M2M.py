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
from utils import *
import pickle
from gensim.models import word2vec
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
# enc = OneHotEncoder()

# import logging_MODEL_FILE
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Reading the input Google word2vec model
#model_vec = word2vec.Word2Vec.load_word2vec_format('/Users/ScrmBison/Desktop/GoogleNews-vectors-negative300.bin', binary=True)
#model_vec = word2vec.Word2Vec.load_word2vec_format('/Users/Rouzbeh/BoxSync/Fall2016/ESCALES/GoogleNews-vectors-negative300.bin', binary=True)
#_VECTOR_SIZE = int(os.environ.get('VECTOR_SIZE', '300'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '400'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.00625'))
_NEPOCH = int(os.environ.get('NEPOCH', '30'))
evaluate_loss_after = 2
# I just want to have all of the word tokens that's why I used 10823 here
vocabulary_size = 3000

_MODEL_FILE = os.environ.get('MODEL_FILE')
# vector_size = _VECTOR_SIZE
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
# enc.fit(feat_1hot)
# for elems in feat_vec:
#     for indix, val in enumerate(elems):
#         elems[indix] = enc.transform(val).toarray().T
# pickle.dump(feat_vec, open("feat_vec.p", "wb"))
feat_vec = pickle.load(open("feat_vec.p", 'rb'))


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

vector_size = vocabulary_size + features2

class LSTMNumpy:
     
    def __init__(self, vector_dim ,label_dim = 2, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.vector_dim = vector_dim
        # Randomly initialize the network parameters
        self.u = np.random.uniform(-np.sqrt(1./vector_dim), np.sqrt(1./vector_dim), (hidden_dim, vector_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (label_dim, hidden_dim))
        self.w = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        # First I initialized u and w in the previous lines and now I am concatenating them into one matrix
        # so that I don't have to deal with two matrices
        self.p = np.concatenate((self.u,self.w) , axis = 1)
        self.W = np.concatenate((self.p, self.p, self.p, self.p) , axis = 0)

def forward_propagation(self, x, feat_vec_train):
    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s because need them later.
    # We add one additional element for the initial hidden, which we set to 0
    s = np.zeros((T + 1, self.hidden_dim))
    c = np.zeros((T + 1, self.hidden_dim))
    # Here I am considering z as summation of the size of input plus hidden layer
    z = np.zeros((T, 4 * self.hidden_dim))
    a = np.zeros((T, self.hidden_dim))
    i = np.zeros((T, self.hidden_dim))
    f = np.zeros((T, self.hidden_dim))
    # remember this oo is different from o, oo is the output gate and o is the final output that we have
    oo = np.zeros((T, self.hidden_dim))

    s[-1] = np.zeros(self.hidden_dim)
    c[-1] = np.zeros(self.hidden_dim)
    # The outputs at each time step. Again, we save them for later.
    o = np.zeros((T, self.label_dim))
    # For each time step...
    for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        X_i = np.zeros(self.vector_dim - 24)
        X_i[x[t]] = 1
        X_i = np.concatenate((X_i, feat_vec_train[t].reshape((-1,)) ), axis = 0)
        # putting x[t] and s[t-1] in one vector and call it I
        I = np.concatenate((X_i,s[t-1]),axis = 0)
        z[t] = self.W.dot(I)
        a[t] = np.tanh(z[t][:self.hidden_dim])
        i[t] = sigmoid(z[t][self.hidden_dim:2*self.hidden_dim])
        f[t] = sigmoid(z[t][2*self.hidden_dim:3*self.hidden_dim])
        oo[t] = sigmoid(z[t][3*self.hidden_dim:])
        c[t] = (i[t] * a[t]) + (f[t] * c[t-1])
        s[t] = oo[t] * (np.tanh(c[t]))
        o[t] = softmax(self.V.dot(s[t]))
    return [o, s, c, z, a, i, f, oo]
LSTMNumpy.forward_propagation = forward_propagation

def predict(self, x, feat_vec_train):
    # Perform forward propagation and return index of the highest score
    o, s = self.forward_propagation(x, feat_vec_train)[:2]
    return np.argmax(o, axis=0)
 
LSTMNumpy.predict = predict


def calculate_total_loss(self, x, y, feat_vec_train):
    L = 0.0
    # For each sentence...
    for i in np.arange(len(y)):
        o, s = self.forward_propagation(x[i], feat_vec_train[i])[:2]
        # We only care about our prediction of the "correct" words
        correct_word_predictions = o[y[i]]
        # Add to the loss based on how off we were
        L += -1.0 * (np.log(correct_word_predictions))
    return L
 
def calculate_loss(self, x, y, feat_vec_train):
    # Divide the total loss by the number of training examples
    # y -> all y_train[i]
    N = len(y)
    return self.calculate_total_loss(x,y, feat_vec_train)/N
 
LSTMNumpy.calculate_total_loss = calculate_total_loss
LSTMNumpy.calculate_loss = calculate_loss

def bptt(self, x, y, feat_vec_train):
    T = len(x)
    # Perform forward propagation
    o, s, c, z, a, i, f, oo = self.forward_propagation(x, feat_vec_train)
    # We accumulate the gradients in these variables
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    # since we are just going self.bptt_truncate steps backward
    dLdc_t = np.zeros((T,self.hidden_dim))
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.
    dLds_t = np.zeros((T + 1, self.hidden_dim))
    # For each output backwards...
    # considering the last element of the hidden layer which
    # is s[T-1], pay attention that s[T] or s[-1] is for the
    # the input of the first layer
    # dLdV += np.outer(delta_o, s[T-1].T)
    # Initial delta calculation
    # dLds_t = self.V.T.dot(delta_o)
    
    # Backpropagation through time (for at most self.bptt_truncate steps)
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)
        dLds_t[t] += self.V.T.dot(delta_o[t])
        for bptt_step in np.arange(max(0, (T-1)-self.bptt_truncate), (T-1)+1)[::-1]:
            #print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            #delta_t = dLds_t * (1 - (s[bptt_step] ** 2))
            dLdo_t = dLds_t[bptt_step] * np.tanh(c[bptt_step])
            dLdc_t[bptt_step] +=  dLds_t * oo[bptt_step] * (1 - (np.tanh(c[bptt_step]))**2)
            dLdi_t = dLdc_t[bptt_step] * a[bptt_step]
            dLdf_t = dLdc_t[bptt_step] * c[bptt_step - 1]
            dLda_t = dLdc_t[bptt_step] * i[bptt_step]
            dLdc_t[bptt_step-1] = dLdc_t[bptt_step] * f[bptt_step]
            # this one is for derivative with respect to a hat
            dLdA_t = dLda_t * (1 - (a[bptt_step])**2)
            dLdI_t = dLdi_t * i[bptt_step] * (1 - i[bptt_step])
            dLdF_t = dLdf_t * f[bptt_step] * (1 - f[bptt_step])
            dLdO_t = dLdo_t * oo[bptt_step] * (1 - oo[bptt_step])
            dLdz_t = np.concatenate((dLdA_t, dLdI_t, dLdF_t, dLdO_t), axis = 0)
            X_j = np.zeros(self.vector_dim - 24)
            X_j[x[bptt_step]] = 1
            X_j = np.concatenate((X_j, feat_vec_train.reshape((-1,)) ), axis = 0)
            # putting x[t] and s[t-1] in one vector and call it I
            I = np.concatenate((X_j,s[bptt_step-1]),axis = 0)
            # Update delta for next step
            # This is for derivative with respect s[t-1], I am just taking the elements that are related to it
            # other elements are for X which we don't need them
            dLds_t[bptt_step - 1] += (self.W.T.dot(dLdz_t))[self.vector_dim:]
            # Updating the weights
            dLdW += np.outer(dLdz_t, I)   
    return [dLdV, dLdW]


LSTMNumpy.bptt = bptt


# Performs one step of SGD.
def numpy_sgd_step(self, x, y, feat_vec_train, learning_rate):
    # Calculate the gradients
    dLdV, dLdW = self.bptt(x, y, feat_vec_train)
    # Change parameters according to gradients and learning rate
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW
 
LSTMNumpy.sgd_step = numpy_sgd_step





# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, feat_vec_train, learning_rate, nepoch, evaluate_loss_after):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    print "There is going to be ", nepoch, " epoches"
    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train, feat_vec_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # For each training example...
        
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
    filename11 = 'numpy_model.sav'
    pickle.dump(model, open(filename11, 'wb'))


model = LSTMNumpy(vector_size,hidden_dim)
train_with_sgd(model, X_train, y_train, feat_vec_train, learning_rate, nepoch, evaluate_loss_after)



def predict_label(model, X_test,y_test,feat_vec_test):
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

# filename11 = 'numpy_model.sav'
# model = pickle.load(open(filename11, 'rb'))
print " precision, recall, fscore, accuracy = " , predict_label(model, X_test,y_test,feat_vec_test), "Test Data"
print " precision, recall, fscore, accuracy = " , predict_label(model, X_train,y_train,feat_vec_train), "Train Data"
# tokenized_sentences

print 'nepoch = ',nepoch ,'vocabulary_size = ' ,vocabulary_size ,'hidden_dim = ' ,hidden_dim ,'learning_rate = ',learning_rate


