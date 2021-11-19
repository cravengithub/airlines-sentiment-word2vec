from os import system
import time as tm
import numpy as np
import pandas as pd
import airline_util as au
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import *
from sklearn.svm import SVC
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest


system('clear')
file = 'dataset/Tweets.csv'
dataFrame = pd.read_csv(file)
corpus = dataFrame.get('text')
sentiments = np.array(dataFrame.get('airline_sentiment'))

# preprocessing text
print('runing preprocessing text', 20 * '=')
clean_text_list = au.preprocessing(corpus)

# making bag of word using vectorization
np.set_printoptions(precision=2)
docs = np.array(clean_text_list)
count = CountVectorizer()
bag = count.fit_transform(docs)

# give weight using tf idf and extract features
tfidf = TfidfTransformer(smooth_idf=False)
features = tfidf.fit_transform(bag)
print('feature shape: ', features.shape)


bayes = MultinomialNB()
start = tm.time()
bayes.fit(features, sentiments)
end = tm.time()
print(bayes)
scores = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']
# clf = AdaBoostClassifier(base_estimator=MultinomialNB(), random_state=0)
for sc in scores:
    val = cross_val_score(bayes, features, sentiments,
                          cv=5, scoring=sc, n_jobs=-1).mean()
    print(sc, ': ', ' %0.5f' % val)
print('training time: ', (end-start))
# new_feature = SelectKBest(
#     mutual_info_classif, k=10).fit_transform(features, label)
# mi = mutual_info_classif(features, label
'''

# print(new_feature.toarray())
rst = mutual_info_classif(new_feature, label)

i =1
for score, fname in sorted(zip(rst, label), reverse=True):
    print(i,',',fname, ',', round(score, 5))
    i+=1
'''


'''
# declare classifier and training data
bayes = MultinomialNB()
C = 1.0
svm_linear = SVC(kernel='linear', C=C)
svm_rbf = SVC(kernel='rbf', gamma=0.7, C=C)
svm_poly = SVC(kernel='poly', degree=3,  gamma='auto', C=C)
clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=5000)
tr = tree.DecisionTreeClassifier()
'''

'''
# gnb = GaussianNB(






nbrs = NearestNeighbors()
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15,), random_state=1)
'''

# model = bayes.fit(features, np.array(label))
# model = svm_rbf.fit(features, np.array(label))
# end = tm.time()
# print('end: ', end)
# print('time: ', round(end-start, 5))

# #evaluation using cross validation
# scores = cross_val_score(bayes, new_feature, label, cv = 5)
# scores = cross_val_score(svm_rbf, features, label, cv=5, n_jobs=-1)
# print(scores)
"""
classifier = [bayes, clf, tr, svm_linear, svm_rbf, svm_poly]
clf_name = ['Naive Bayes', 'Logistic Regression',
            'Decision Tree', 'SVM Linear', 'SVM RBF', 'SVM Polynomial']
scores = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']
for clfr, name in zip(classifier, clf_name):
    print('name: ', name, 50*'=')
    # print('start: ', start)
    start = tm.time()
    for sc in scores:
        val = cross_val_score(clfr, features, label,
                              cv=5, scoring=sc, n_jobs=-1).mean()
        print(sc, ' %0.5f' % val)
    # print('end: ', end)
    end = tm.time()
    print('time: ', round(end-start, 5))
"""

# print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

'''
for i in range(0,101):
    print(clean_text_training_list[i])
print('length: ', len(clean_text_training_list))
'''

'''
kf = KFold(n_splits=3,random_state=42,shuffle=True)
accuracy = []
for train_index, test_index in kf.split(features):
    x_train = features[train_index]
    y_train = label[train_index]
    x_test = features[test_index]
    y_test = label[test_index]

    bayes = MultinomialNB()
    bayes.fit(x_train, y_train)
    predict = bayes.predict(x_test)
    acc = accuracy_score(y_test, predict)
    accuracy.append(acc)
avg = np.mean(accuracy)
print(avg)
'''


'''
print(str(corpus[0]))
rs = au.tokenizer(str(corpus[0]))
print(rs)
'''

'''
#feature selection using mutual information
rst = mutual_info_classif(features, label)
for score, fname in sorted(zip(rst, label), reverse=True):
    print(fname, score)
'''
