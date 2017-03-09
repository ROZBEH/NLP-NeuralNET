#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano
import pickle

_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')
vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading CSV file..."
with open('data/reddit-comments-2015-08.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

print "Parsed %d sentences." % (len(sentences))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

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
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
print y_train[0]
print X_train[0]


'''
	1. find the order and initialize Us and Ws accordingly
				f    :       W    if previous case: dim [0 = x]
							 U
				o    :       W
				             U
				g    :       W
				             U
				i    :       W
				             U

		T   =     len(x)   // x is the input sentence, which is the no. units of the
						   // network
		i   =     np.zeros(T, size)

		for sure     :      (
							 np.zeros(T, dimensions of c) => coz c(t) = c(t-1) .* f + g .* i 
							 np.zeros(T, dimensions of s) => coz s(t) = tanh(c(t)) .* o
							 				s -> dimension 1 X 100 in previous cases         ,



		z_t1   =    U.x_e   +   W.s(t-1)



'''

import math
import numpy as np

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '2000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '100'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_LABEL_DIM = int(os.environ.get('LABEL_DIM', '6'))
_WORD_DIM = int(os.environ.get('WORD_DIM', '6'))
v_s = _VOCABULARY_SIZE


'''
		x -> 2000 x 1 (Ux) -> 100 x 1
		U -> 100 x 2000
		s -> 100 x 1 (Ws) -> 100 x 1
		W -> 100 x 100
		i  ,  f  ,  o  -> 100 x 1 -> np.zeros(T, 100 x 1) -> during initialization
		g -> 100 x 1
		c -> 100 x 1

		dldU_i

'''

class RNNNumpy:

	'''
		0   -   f
		1   -   o
		2   -   g
		3   -   i
	'''

	def __init__(self, hidden_dim, word_dim, label_dim, bptt_truncate):
		self.label_dim = label_dim
		self.hidden_dim = hidden_dim
		self.word_dim = word_dim
		self.bptt_truncate = bptt_truncate
		self.W = np.random.uniform(-np.sqrt(1./vector_dim), np.sqrt(1./vector_dim), (4, hidden_dim, hidden_dim))
		self.U = np.random.uniform(-np.sqrt(1./vector_dim), np.sqrt(1./vector_dim), (4, hidden_dim, word_dim))
		self.V = np.random.uniform(-np.sqrt(1./vector_dim), np.sqrt(1./vector_dim), (2000, 100))
		
# e**j /

# create a general function for derivative calculation to which you can just 


def hard_softmax(k):
	val = []
	l = len(k)
	j = sum([np.exp(k[i]) for i in range(l)])
	val = [np.exp(k[i])/j for i in range(l)]
	return val

def forward_propagation(self, x):
	T = len(x)
	c = np.zeros((T, dim_c))
	s = np.zeros((T+1, dim_s))
	i = np.zeros((T, dim_i))
	# v -> 100 x 8000
	# x -> order of x -> sth x hidden_dim
	# 
	for t in range(T):
		c[t] = np.multiply(c[t-1], f) + np.multiply(g, i)
		s[t] = np.multiply(np.tanh(c[t]), o)
		f[t] = hard_softmax(self.U[0].dot(x[t])+s[t-1].dot(W[0]))
		i[t] = hard_softmax(self.U[3].dot())

def bptt(self, input_sentence):
	T = len(input_sentence)
	o, s = self.forward_propagation(input_sentence)
	dLdV = np.zeros(self.V.shape)
	for i in range(T):


	









































