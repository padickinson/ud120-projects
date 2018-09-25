#!/usr/bin/python

import sys
import pickle
import os
import numpy as np

TOOLSPATH = '../tools/'
DATAPATH = './'

if os.getcwd().split('/')[-1] != 'final_project':
    TOOLSPATH = './tools/'
    DATAPATH = './final_project/'
sys.path.append(TOOLSPATH)

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### Load the dictionary containing the dataset
with open(DATAPATH + "final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Remove "Total" row from data
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
    ### label
    'poi',

    ### payments
    # 'salary',
    # 'bonus',
    # 'long_term_incentive',
    # 'deferred_income',
    # 'deferral_payments',
    # 'loan_advances',
    # 'other',
    # 'expenses',
    # 'director_fees',
    # 'total_payments',

    ### stock value
    # 'exercised_stock_options',
    # 'restricted_stock',
    # 'restricted_stock_deferred',
    # 'total_stock_value',

    ### emails to/from POI
    # 'to_messages',
    'from_messages',
    # 'shared_receipt_with_poi',
    'from_this_person_to_poi',
    # 'from_poi_to_this_person',
    ]


raw_features = np.array([[float(data_dict[name][f]) for f in features_list] for name in data_dict.keys()])
from understand_the_data import summarize_data
summarize_data(features_list[1:], raw_features[:,0], raw_features[:,1:])

### Task 2: Remove outliers

# show histograms of data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

# nulls = np.isnan(raw_features[:,3])
# X = raw_features[:,3][~nulls]
# print(X)
# plt.hist(X)
# plt.show()

### Look at correlation between who sends mails to and receives mails from POI
names = data_dict.keys()
poi = np.array([data_dict[name]['poi'] for name in names])
mails_to_poi = np.array([float(data_dict[name]['from_this_person_to_poi']) for name in names])
mails_from_poi = np.array([float(data_dict[name]['from_poi_to_this_person']) for name in names])
plt.scatter(mails_to_poi[~poi],mails_from_poi[~poi], color='b')
plt.xlabel('from_this_person_to_poi')
plt.ylabel('from_poi_to_this_person')
plt.scatter(mails_to_poi[poi],mails_from_poi[poi], color='r')
# plt.show()

# Validate that the big senders look real
for name in names:
    if float(data_dict[name]['from_poi_to_this_person']) > 300:
        pass
        # print (name,data_dict[name])


### Task 3: Create new feature(s)

names = np.array(data_dict.keys())

### 1. pct_from_this_person_to_poi
mails_from_this_person = np.array([float(data_dict[name]['from_messages']) for name in names])
pct_from_this_person_to_poi = mails_to_poi / mails_from_this_person
for ii in range(len(names)):
    data_dict[names[ii]]['pct_from_this_person_to_poi'] = \
        pct_from_this_person_to_poi[ii] if np.isfinite(pct_from_this_person_to_poi[ii]) else 0
features_list.append('pct_from_this_person_to_poi')


### 2. pct_to_this_person_from_poi
mails_to_this_person = np.array([float(data_dict[name]['to_messages']) for name in names])
pct_to_this_person_from_poi = mails_from_poi / mails_to_this_person
for ii in range(len(names)):
    data_dict[names[ii]]['pct_to_this_person_from_poi'] = \
        pct_to_this_person_from_poi[ii] if np.isfinite(pct_to_this_person_from_poi[ii]) else 0
features_list.append('pct_to_this_person_from_poi')

### 3. person_id
id_lookup = {}
for ii in range(len(names)):
    data_dict[names[ii]]['person_id'] = ii
    id_lookup[ii] = names[ii]
features_list.append('person_id')


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
# NB: This changes the order of rows, so you can't lookup with names
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

for i in range(len(features)):
    print(id_lookup[features[i][-1]],labels[i],features[i])

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#
# from sklearn.svm import SVC
# clf = SVC(kernel="rbf", C=10000.0)
#
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier()



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
# print(features_train)
pred = clf.predict(features_test)

from sklearn.metrics import precision_score, recall_score
print 'Precision: {precision:0.2f} Recall {recall:0.2f}:  '.format(
    precision=precision_score(labels_test, pred),
    recall=recall_score(labels_test, pred)
    )


# test_classifier(clf,)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
