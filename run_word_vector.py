import airline_util as au
from os import system
from gensim.models import *
from pandas import Series
from pandas import DataFrame
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

system('clear')
# obtain corpus
file = './dataset/Tweets.csv'
corpus = au.get_corpus(file, 'text')
label = np.array(au.get_corpus(file, 'airline_sentiment'))
# print(len(label))

# preprocessing text
print('runing preprocessing text', 20 * '=')
clean_text_list = au.preprocessing(corpus)

# split tweet into tokens
print('split tweet into tokens', 20 * '=')
tokens = []
for ct in clean_text_list:
    tokens.append(ct.split())
# print(tokens[0])

# print(label.shape)

# training word vector
print('training word vector', 20 * '=')

data = Series(tokens).values
vector_size = 200
# phrases = Phrases(data)
model = Word2Vec(data, min_count=1, vector_size=vector_size,
                 window=3, workers=3, sg=1)
print(model)
# build data training for classifier
print('build data training for classifier', 20 * '=')
i = 0
data_training = []
with open('./dataset/data_training.csv', 'wt',) as f:
    for tk in tokens:
        tmp = None
        print(i, ': ', ' length: ', len(tk))
        if len(tk) == 0:
            tmp = np.zeros([1, vector_size], dtype=np.int8)
        else:
            res = np.array(model.wv[tk]).mean(axis=0, dtype=np.float_)
            tmp = res.reshape(1, vector_size)
        # print(i, ': ', tk)
        # print(tmp)

        list = [str(e) for e in tmp.tolist()]
        line = ','.join(list)
        line = line.replace('[', '')
        line = line.replace(']', '') + '\n'
        # line = line[0:len(line)-1]
        f.writelines(line)
        # print(line)
        # print()
        # data_training.append(tmp.tolist())
        # if i == 800:
        # break
        i += 1
    f.close()

# data_frame = DataFrame(data=data_training)
# data_frame.to_csv('./dataset/data_training.csv', header=False, index=False,
#                   doublequote=False,
#                   escapechar=' ')


"""
# Evaluation using Naive Bayes
print("Running Evaluation.....", 20*'=')
clf = MultinomialNB()
scores = cross_val_score(clf,data_training, label, cv=5,
                         n_jobs=-1, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
"""

"""
# build Bag of Words
bag = au.bow(clean_text_list)
# print(ls)

# give weight using tf idf and extract features
tfidf = TfidfTransformer()
features = tfidf.fit_transform(bag)
# nb = MultinomialNB()
clfr = AdaBoostClassifier(base_estimator=nb, n_estimators=300,random_state=0)

# clfr = BaggingClassifier()
scores = cross_val_score(clfr,features, label,
                         scoring='accuracy',cv=3, n_jobs=-1)

print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
# print('feature shape: ', features.shape)
# print(features)
"""
