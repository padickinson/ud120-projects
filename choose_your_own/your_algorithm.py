import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from  sklearn.metrics import accuracy_score
features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary
features_train, labels_train, features_test, labels_test = makeTerrainData()

results = {}
###################### Naive Bayes #############################
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print('Naive Bayes: ', accuracy_score(pred,labels_test))

########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(kernel="linear")
# clf = SVC(kernel="rbf", C=1, gamma=100)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print('SVC -- Linear: ', accuracy_score(pred,labels_test))

# clf = SVC(kernel="linear")
clf = SVC(kernel="rbf", C=1, gamma=150)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print('SVC -- RBF: ', accuracy_score(pred,labels_test))


########################## Decision Tree #################################
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=2)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print('DT samples=2: ', accuracy_score(pred,labels_test))


clf = DecisionTreeClassifier(min_samples_split=50)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred,labels_test)
print('DT samples=50: ', accuracy_score(pred,labels_test))

###################### k-Nearest Neighbour #############################
from sklearn.neighbors import KNeighborsClassifier
for i in [2,5,10,15,19,20,21,25,30,40,50]:
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    acc = accuracy_score(pred,labels_test)
    print(i,'-NN: ',acc)
#### Results
#  k : accuracy_score
#  2 : .928
#  5 : .92
# 10 : .932
# 20 : .936
# 50 : .928

# ###################### AdaBoost #############################
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10),n_estimators=50,learning_rate=1)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred,labels_test)
print('AdaBoost: ',acc)

###################### Random Forest #############################
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=15)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred,labels_test)
print('Random Forest n=15: ',acc)

# 0.92, 0.928, 924
#
# try:
#     prettyPicture(clf, features_test, labels_test)
# except NameError:
#     pass
