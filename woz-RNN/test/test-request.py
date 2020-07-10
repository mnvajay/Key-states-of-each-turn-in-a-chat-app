import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import json
import codecs
import pickle
from keras.preprocessing.text import one_hot
from keras.layers import Activation
from numpy import array
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate

with open('../tokenization/tokenizer.pickle','rb') as handle:
   tokenizer = pickle.load(handle)

word_index = tokenizer.word_index
file_path = "request-option1-test.json"

data = json.load(codecs.open(file_path,"r","utf-8"))

vocab = []
for idx in range(0,len(data)):
   vocab = vocab + data[idx]['dialog'].split()

vocab_size = len(set(vocab))

test_data = []
test_labels = []

for idx in range(0,len(data)):
    test_data.append(data[idx]['dialog'])
    test_labels.append(data[idx]['slots'])

test_labels = array(test_labels)

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit([['address','area' ,'food', 'name', 'phone', 'postcode', 'price range']])

labels = mlb.transform(test_labels)
##labels = np.asarray(labels).astype(np.float32)

##encoded_data = [one_hot(d,vocab_size) for d in test_data]
##print(encoded_data)

encoded_data = [[word_index[i] for i in sent.replace('?',' ').split()] for sent in test_data]

test_data = keras.preprocessing.sequence.pad_sequences(encoded_data,value=0,padding="post",maxlen=51,dtype=object)
print(test_data.shape)

model = keras.models.load_model("../training/request.h5")
y_pred = model.predict(test_data)
print(y_pred[3])
y_pred = np.where(y_pred > 0.5, 1, 0)
##print(y_pred[3])

score = model.evaluate(test_data, labels, verbose=0)
##print(84.2644975556)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
##print(accuracy_score(y_pred,labels))


