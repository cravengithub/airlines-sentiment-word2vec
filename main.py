import os

import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

import airline_util as au

os.system('cls')

porter = PorterStemmer()

# obtain corpus
corpus = au.get_corpus()

clean_text_training_list = []
for text in corpus:
    clean_text_training = ''
    # tokenizer corpus and remove letter sign
    for s in au.tokenizer(text):
        # stemming word and stopwords removal
        clean_text_training = clean_text_training + porter.stem(s) + ' '
    clean_text_training_list.append(clean_text_training[0:len(clean_text_training) - 1])

# making bag of word using vectorization
np.set_printoptions(precision=2)
docs = np.array(clean_text_training_list)
count = CountVectorizer()
bag = count.fit_transform(docs)

# define label
label = [1, 0, 0, 1]

# give weight using tf idf and extract features
# tfidfVec = TfidfVectorizer(stop_words='english')
tfidf = TfidfTransformer(smooth_idf=False)
features = tfidf.fit_transform(bag)
# features = tfidfVec.fit_transform(clean_text_training_list)
# print(features.toarray())

# declare classifier and training data
bayes = MultinomialNB()
model = bayes.fit(features, np.array(label))
# print(model)


text_test = 'the sun is so hot'
clean_text_test_list = []
clean_text_test = ''
for s in au.tokenizer(text_test):
    # stemming word and stopwords removal
    clean_text_test = clean_text_test + porter.stem(s) + ' '
clean_text_test_list.append(clean_text_test[0:len(clean_text_test) - 1])


test_bag = count.transform(np.array(clean_text_test_list))
test_features = tfidf.transform(test_bag)
# test_features = tfidfVec.fit_transform(clean_test_list)
# print(test_features.toarray())
# print(test_features.shape)

#feature selection using mutual information
'''
rst = mutual_info_classif(features, label)

for score, fname in sorted(zip(rst, label), reverse=True):
    print(fname, score)
'''

# evalutain using cross valitdation
'''
predicted = bayes.predict(test_features)
scores = cross_val_score(bayes, features, label, cv = 2)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)
print(predicted)
'''



