#!/usr/bin/python


from time import time
from test import getData
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

t0 = time()
features_train, features_test, labels_train, labels_test = getData()
print "Data retrieval time: ", round(time()-t0, 3), "s"



# param_grid ={
# 				'kernel': ['rbf'], 
# 				'gamma': [0.0001, 0.01, 0.1],
#                 'C': [175, 210, 300, 250]
#              }

### best from c= 210 and kernel= rbf and gamma = 0.1
 
svm = SVC(C=210, kernel = 'rbf', gamma = 0.1)
t0 = time()
#clf = GridSearchCV(svm, param_grid)
clf = svm
clf.fit(features_train, labels_train)
print "GridSearch time: ", round(time()-t0, 3), "s"

#print "Best estimator found by grid search:"
#print clf.best_estimator_

pred = clf.predict(features_test)

accuracy = accuracy_score(labels_test, pred)

print accuracy, "\n"

