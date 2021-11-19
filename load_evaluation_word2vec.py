import os
import numpy as np
import pandas as pd
from gensim.models import *
from nltk.stem.porter import PorterStemmer
import time as tm

from sklearn.naive_bayes import *
from sklearn.svm import SVC
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import airline_util as au

os.system('clear')

porter = PorterStemmer()

# obtain corpus
# corpus = au.get_corpus()

# obtain corpus
file = './dataset/Tweets.csv'
file_out = './dataset/model.npy'
file_out2 = './dataset/label.npy'
dataFrame = pd.read_csv(file)
corpus = dataFrame.get('text')
sentiments = np.array(dataFrame.get('airline_sentiment'))
data_train = np.load(file_out)
label = np.load(file_out2)
print(data_train.shape)
print(label.shape)

# Evaluation section ===========================================================================
bayes = GaussianNB()
clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=5000)
tr = tree.DecisionTreeClassifier()
C = 1.0
svm_linear = SVC(kernel='linear', C=C)
svm_rbf = SVC(kernel='rbf', gamma=0.7, C=C)
svm_poly = SVC(kernel='poly', degree=3,  gamma='auto', C=C)

# classifier = [bayes, clf, tr, svm_linear, svm_rbf, svm_poly]
classifier = [svm_linear, svm_rbf, svm_poly]
# clf_name = ['Naive Bayes', 'Logistic Regression',
#             'Decision Tree', 'SVM Linear', 'SVM RBF', 'SVM Polynomial']
clf_name = ['SVM Linear', 'SVM RBF', 'SVM Polynomial']
scores = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']
for clfr, name in zip(classifier, clf_name):
    print('\nname: ', name, 50*'=')
    # print('start: ', start)
    start = tm.time()
    for sc in scores:
        val = cross_val_score(clfr, data_train, label.ravel(order='C'),
                              cv=5, scoring=sc, n_jobs=-1).mean()
        print(sc, ' %0.5f' % val)
    # print('end: ', end)
    end = tm.time()
    print('time: ', round(end-start, 5))
