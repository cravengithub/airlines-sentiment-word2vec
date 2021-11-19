from scipy.sparse import data
import airline_util as au
from os import system
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import time as tm

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

system('clear')

# obtain corpus
file = './dataset/Tweets.csv'
data_training_file = './dataset/data_training.csv'
label = np.array(au.get_corpus(file, 'airline_sentiment'))
dtn = pd.read_csv(data_training_file, header=None, low_memory=False)
data_training_raw = np.array(dtn)
scaler = MinMaxScaler()
scaler.fit(data_training_raw)
data_training = scaler.transform(data_training_raw)
# label = label.reshape([len(label), 1])
print(label.shape)
print(data_training.shape)
# for lb in label:
# print(lb)
# Evaluation using Naive Bayes
print("Running Evaluation.....", 20*'=')
# clf = DecisionTreeClassifier()
# clf = MultinomialNB()
# clf = AdaBoostClassifier(base_estimator=MultinomialNB(), random_state=0)
clf = BaggingClassifier(base_estimator=MultinomialNB(), random_state=0)
start = tm.time()
clf.fit(data_training, label)
end = tm.time()
print(clf)
print('name: ', 'Naive Bayes + Bagging')
scores = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']
for sc in scores:
    val = cross_val_score(clf, data_training, label,
                          cv=5, scoring=sc, n_jobs=-1).mean()
    print(sc, ': ', ' %0.5f' % val)
print('training time: ', (end-start))

# clf = AdaBoostClassifier(base_estimator=SVC(
#     kernel='linear', C=1.0), algorithm="SAMME")
# clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
#                          random_state=0)
# clf = SVC(kernel='linear', C=1.0)
# clf.fit(data_training, label)

# scores = cross_val_score(clf, data_training, label,
#  cv=5, n_jobs=-1, scoring='accuracy')
# print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
