import os
import numpy as np
import pandas as pd
from gensim.models import *
from nltk.stem.porter import PorterStemmer
import time as tm

from sklearn.naive_bayes import *
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
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


clean_text_training_list = []
# tokens = []
for text in corpus:
    clean_text_training = ''
    # tokenizer corpus and remove letter sign
    for s in au.tokenizer(text):
        # stemming word and stopwords removal
        clean_text_training = clean_text_training + porter.stem(s) + ' '
        # tokens.append(porter.stem(s))
        # print(clean_text_training)
    string_tmp = clean_text_training[0:len(clean_text_training) - 1]
    clean_text_training_list.append(string_tmp.split())

# define label
label = []
for lb in sentiments:
    label.append(lb)

model = Word2Vec(size=300, min_count=1, workers=4)
# building vocabulary for training
model.build_vocab(clean_text_training_list)
print("\n Training the word2vec model...\n")
# reducing the epochs will decrease the computation time
model.train(clean_text_training_list, total_examples=len(
    clean_text_training_list), epochs=4000)

data_train = []
data_class = []
i = 0
for train, lb in zip(clean_text_training_list, label):
    if len(train) != 0:
        row, col = model[train].shape
        arr = model[train]
        new_arr = np.ones(col)
        np.put(new_arr, [0], [23])
        for c in range(0, col):
            np.put(new_arr, [c], [np.mean(arr[:, c])])
        data_train.append(new_arr)
        data_class.append([lb])
        i += 1
        print('===-', i)
        # break

data_train = np.array(data_train)
data_class = np.array(data_class)
np.save(file_out, data_train)
np.save(file_out2, data_class)

print('data train shape: ', data_train.shape)
print('data class shape: ', data_train.shape)

# for dtt in data_train:
#     print(dtt.shape)
print(data_train.shape)
print(data_class.shape)
