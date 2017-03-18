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
import logging



_VECTOR_SIZE = int(os.environ.get('VECTOR_SIZE', '300'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '400'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.1'))
_NEPOCH = int(os.environ.get('NEPOCH', '10'))
evaluate_loss_after = 2
vocabulary_size = 5000

_MODEL_FILE = os.environ.get('MODEL_FILE')
vector_size = _VECTOR_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
hidden_dim = _HIDDEN_DIM
learning_rate = _LEARNING_RATE
nepoch = _NEPOCH


class RNNNumpy:
     
    def __init__(self, vector_dim, hidden_dim=100, label_dim = 6, bptt_truncate=4):
        # Assign instance variables
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.vector_dim = vector_dim
    def printing(self):
        print self.label_dim
        print self.hidden_dim
        print self.bptt_truncate
        print self.vector_dim

model = RNNNumpy(vector_size,hidden_dim)
model.printing()



