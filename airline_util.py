
import re
import pandas as pd
import  numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# obtain corpus
def get_corpus(file, attribute=''):
    corpus = None
    '''
    corpus = [
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining and the weather is sweet',
    'Is that good news to day of this sun?']
    '''
    if file is not None:
        data_frame = pd.read_csv(file)
        corpus = data_frame.get(attribute)
    return corpus

# Tokenizing teks
def tokenizer(text):
    stop = stopwords.words('english')
    text = re.sub('<[^>]*>', '', text)
    hashTagRemover = [w for w in text.split() if not w.startswith('#') ]
    retweetRemover = [w for w in hashTagRemover if not w.startswith('@') ]
    str = ' '.join(retweetRemover)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', str.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


# preprocessing text
def preprocessing(corpus):
    porter = PorterStemmer()
    clean_text_training_list = []
    for text in corpus:
        clean_text_training = ''
        # tokenizer corpus, stopwords removal, and remove letter sign
        for s in tokenizer(str(text)):
            # stemming word
            clean_text_training = clean_text_training + porter.stem(s) + ' '
            # clean_text_training = clean_text_training + s +' '
        clean_text_training_list.append(
            clean_text_training[0:len(clean_text_training)])
    return clean_text_training_list

# build Bag Of Word (bow)
def bow(clean_text_list=[]):
    bag = None
    np.set_printoptions(precision=2)
    docs = np.array(clean_text_list)
    count = CountVectorizer()
    bag = count.fit_transform(docs)
    return  bag