from gensim.models import Word2Vec
from gensim import downloader
from os import system
from pandas import Series
import numpy as np

system('clear')
sentences = [['saya', 'makan', 'bakso', 'di', 'warung', 'hari', 'ini']]
# data = Series(np.array(sentences)).values
# print(data)
model = Word2Vec(sentences, vector_size=5, min_count=1, workers=4, window=3)
print(model.wv['saya', 'makan'])
line = ",".join(['saya', 'makan', 'bakso'])
print(line)


glove_vector = downloader.load('glove-twitter-100')
print(glove_vector['twitter'].shape)
