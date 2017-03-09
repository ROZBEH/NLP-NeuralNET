import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
import pickle
from sklearn.naive_bayes import GaussianNB
# from bin import *
warnings.filterwarnings("ignore")
all_classes = np.array([0, 1])
X = pickle.load( open( "X_feature_all_tag_count.p", "rb" ) )
y = pickle.load( open( "y_label_all_tag_count.p", "rb" ) )

x_l1 = 0
x_h1 = 0
x_l2 = 0
x_h2 = 0
x_l3 = 0
x_h3 = 0
 # for item in X_feature:
 # 	if item
pos = 0
neg = 0
for item in y:
	if item == 1 :
		pos += 1
	elif item == 0:
		neg += 1
print pos
print neg
print pos + neg 
print len(y)

X_ = np.array(X)
y_ = np.array(y)
X_train = X_[2*len(X)/10:]
y_train = y_[2*len(X)/10:]
X_test = X_[0:2*len(X)/10]
y_test = y_[0:2*len(X)/10]
# logistic = LogisticRegression()
gnb = GaussianNB()
print "starting the classifier"

# logistic.fit(X_train,y_train)
gnb.partial_fit(X_train, y_train, classes=all_classes)
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0 

y_pred = gnb.predict(X_test)

for I in range(y_test.shape[0]):
    if y_pred[I] == y_test[I]:
    	if y_test[I] == 1:
    		true_pos += 1
    	else:
    		true_neg += 1
    else:
    	if y_test[I] == 1:
    		false_neg += 1
    	else:
    		false_pos += 1

print "true pos = ", true_pos
print "true neg = ", true_neg
print "false pos = ", false_pos
print "false neg = ",false_neg
precision = (true_pos)/float(true_pos + false_pos)
recall = (true_pos)/float(true_pos + false_neg)
fscore = (2*precision*recall)/float(precision+recall)
accuracy = (true_neg + true_pos)/float(true_neg + true_pos + false_neg + false_pos)
print "precision = ", precision
print "recall = ", recall
print "accuracy = ", accuracy
print "F1 = ", fscore