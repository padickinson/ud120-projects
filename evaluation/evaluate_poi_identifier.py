#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
import numpy as np
sys.path.append("./tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("./final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

### "MVP" to get something working
### Train and test on same data (overfits!)
# clf = DecisionTreeClassifier()
# clf.fit(features,labels)
# pred = clf.predict(features)
# acc = accuracy_score(labels,pred)
# print(acc)

### Split data into test and train and re-train model
features_train, features_test, labels_train, labels_test = train_test_split(
    features,labels,
    random_state=42, test_size=0.3
)
### Fit and predict Decision tree
clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)


acc = accuracy_score(labels_test,pred)
print(acc)
print(sum(pred))
print(len(pred))

print(precision_score(labels_test,pred))
print(recall_score(labels_test,pred))

# print(np.column_stack([np.array(labels_test).transpose(),pred.transpose()]))
# # print(np.array(np.array(labels),pred))
# print(map(lambda x : x[0]-x[1], np.column_stack([np.array(labels_test).transpose(),pred.transpose()])))
