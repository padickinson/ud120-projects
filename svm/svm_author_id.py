#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("./tools")
print(item.value for item in sys.path)
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()





#########################################################
### your code goes here ###

from sklearn.svm import SVC



# clf = SVC(kernel="linear")


clf = SVC(kernel="rbf", C=10000.0)

#### Reduce training set to speed things up ####
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print ("prediction time:", round(time()-t1, 3), "s")

from sklearn.metrics import accuracy_score
print(accuracy_score(labels_test,pred))


#### Print some predictions ####
# pn = lambda x: print(x, ': ',pred[x])
# pn(10)
# pn(26)
# pn(50)

#### How many "Chris" (i.e. 1) predictions?
# print(sum(pred))
# Value: 877

#### Prediction Accuracy ####
# With full training data: 0.984
# With 1% training data (linear): 0.885
# 1% RBF: 0.616
# 1% RBF, C=10.0: 0.616
# 1% RBF, C=100.0: 0.616
# 1% RBF, C=1000.0: 0.821
# 1% RBF, C=10000.0: 0.892
# 100% RBF, C=10000.0: 0.991

#########################################################
