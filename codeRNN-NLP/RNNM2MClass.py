# Here are dealing with the RNNNUMPY class for the task of prediction
import numpy as np
from utils import *
from datetime import datetime
class RNNNumpy:
     
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        # we have 2 output labels as a result we put 2 here.
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (2, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

def forward_propagation(self, x, feat_vec_train):
    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s because need them later.
    # We add one additional element for the initial hidden, which we set to 0
    s = np.zeros((T + 1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
    # The outputs at each time step. Again, we save them for later.
    #output has two labels as a result we put two here
    o = np.zeros((T, 2))
    # For each time step...
    for t in np.arange(T):
        # this is the second feature vector that I am concatenating to the end of the feature vector
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        A = datetime.now()
        s[t] = np.tanh(self.U[: , x[t]] + self.U[:,self.word_dim-24:].dot(feat_vec_train[t]).reshape((-1,)) + self.W.dot(s[t-1]))
        B = datetime.now()
        print "s[t] time = ", self.timenow(A,B)
        o[t] = softmax(self.V.dot(s[t]))
        A = datetime.now()
        print "o[t] time = ", self.timenow(B,A)
    return [o, s]
RNNNumpy.forward_propagation = forward_propagation


def timenow(self, A, B):
    TT = str(A).split(' ')[1].split(':')
    TT = [int(TT[0]), int(TT[1]), float(TT[2])]
    QQ = str(B).split(' ')[1].split(':')
    QQ = [int(QQ[0]), int(QQ[1]), float(QQ[2])]
    dif = (QQ[0]-TT[0])*3600 + (QQ[1]-TT[1])*60 + (QQ[2]-TT[2])
    return dif

RNNNumpy.timenow = timenow



def predict(self, x,feat_vec_train):
    # Perform forward propagation and return index of the highest score
    o, s = self.forward_propagation(x,feat_vec_train)
    return np.argmax(o, axis=1)
 
RNNNumpy.predict = predict


def calculate_total_loss(self, x, y,feat_vec_train):
    L = 0
    # For each sentence...
    for i in np.arange(len(y)):
        o, s = self.forward_propagation(x[i],feat_vec_train[i])
        # We only care about our prediction of the "correct" words
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        # Add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions))
    return L
 
def calculate_loss(self, x, y, feat_vec_train):
    # Divide the total loss by the number of training examples
    N = np.sum((len(y_i) for y_i in y))
    return self.calculate_total_loss(x, y, feat_vec_train)/N
 
RNNNumpy.calculate_total_loss = calculate_total_loss
RNNNumpy.calculate_loss = calculate_loss

def bptt(self, x, y,feat_vec_train):
    T = len(y)
    # Perform forward propagation
    o, s = self.forward_propagation(x,feat_vec_train)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.
    # For each output backwards...
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)
        # Initial delta calculation
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            dLdW += np.outer(delta_t, s[bptt_step-1])              
            # for the first part of the feature vector which is the word itself
            feat2 = feat_vec_train[bptt_step].reshape((-1,))
            A = datetime.now()
            dLdU[:,x[bptt_step]] += delta_t
            # for the first part of the feature vector which is the language of the word
            dLdU[:,self.word_dim-24:] += np.outer(delta_t, feat2)
            B = datetime.now()
            print "dLdU time = ", self.timenow(A,B)
            # Update delta for next step
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
            A = datetime.now()
            print "delta_t time = ", self.timenow(B,A)
    return [dLdU, dLdV, dLdW]
 
RNNNumpy.bptt = bptt


# Performs one step of SGD.
def numpy_sgd_step(self, x, y,feat_vec_train, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y,feat_vec_train)
    # Change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW
 
RNNNumpy.sgd_step = numpy_sgd_step





