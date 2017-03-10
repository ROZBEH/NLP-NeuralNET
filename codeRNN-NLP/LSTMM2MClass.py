""" 
Here are dealing with the LSTMNUMPY class for the task of prediction
It is the mathematical implementation of LSTM
"""
import numpy as np
from utils import *





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
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        # Add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions))
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
            dLdc_t[bptt_step] +=  dLds_t[bptt_step] * oo[bptt_step] * (1 - (np.tanh(c[bptt_step]))**2)
            dLdi_t = dLdc_t[bptt_step] * a[bptt_step]
            dLdf_t = dLdc_t[bptt_step] * c[bptt_step - 1]
            dLda_t = dLdc_t[bptt_step] * i[bptt_step]
            dLdc_t[bptt_step-1] = dLdc_t[bptt_step] * f[bptt_step]
            # this one is for derivative with respect to A hat
            dLdA_t = dLda_t * (1 - (a[bptt_step])**2)
            dLdI_t = dLdi_t * i[bptt_step] * (1 - i[bptt_step])
            dLdF_t = dLdf_t * f[bptt_step] * (1 - f[bptt_step])
            dLdO_t = dLdo_t * oo[bptt_step] * (1 - oo[bptt_step])
            dLdz_t = np.concatenate((dLdA_t, dLdI_t, dLdF_t, dLdO_t), axis = 0)
            X_j = np.zeros(self.vector_dim - 24)
            X_j[x[bptt_step]] = 1
            X_j = np.concatenate((X_j, feat_vec_train[bptt_step].reshape((-1,)) ), axis = 0)
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