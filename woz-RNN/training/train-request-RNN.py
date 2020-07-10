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

from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, GRU
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate

# seed-fixing
from numpy.random import seed
seed(30)
tf.set_random_seed(30)


with open('../tokenization/tokenizer.pickle','rb') as handle:
   tokenizer = pickle.load(handle)

word_index = tokenizer.word_index
file_path = "request-option1.json"
data = json.load(codecs.open(file_path,"r","utf-8"))

train_data = []
train_labels = []

for idx in range(0,len(data)):
    train_data.append(data[idx]['dialog'])
    train_labels.append(data[idx]['slots'])

train_labels = array(train_labels)

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(train_labels)
labels = np.asarray(labels).astype(np.float32)
##print(mlb.classes_)

##encoded_data = [one_hot(d,vocab_size) for d in train_data]

encoded_data = [[word_index[i] for i in sent.replace('?',' ').split()] for sent in train_data]

##train_data = keras.preprocessing.sequence.pad_sequences(encoded_data,value=0,padding="pre",maxlen=51,dtype=object)
##
##from keras.layers import MaxPool1D,Conv1D 
##deep_inputs = Input(shape=(51,))
##embedding_layer = Embedding(2373,64)(deep_inputs)
##conv_1D = Conv1D(filters=64,kernel_size=3,activation="relu")(embedding_layer)
##drop_out = Dropout(0.2)(conv_1D)
##max_pool = GlobalMaxPooling1D()(drop_out)
##dense_layer = Dense(7,activation='sigmoid')(max_pool)
##model = Model(inputs=[deep_inputs],outputs=dense_layer)
##
##model.compile(optimizer="adam", loss="binary_crossentropy",metrics = ['accuracy'])
##model.summary()

encoded_data = pad_sequences(encoded_data,padding="post",maxlen=51)

import gensim
WORD2VEC_MODEL = "model_1_lac.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_MODEL,binary=True,limit=100000)

embedding_weights = np.zeros((2373,300))
for word,index in word_index.items():
    try:
        embedding_weights[index] = word2vec[word]
    except:
        pass
    
model = Sequential()
embedding_layer = Embedding(2373,300,weights=[embedding_weights],input_length=51,trainable=False)
model.add(embedding_layer)
model.add(GRU(128))
model.add(Dense(7,activation="sigmoid"))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

history = model.fit(encoded_data,labels,batch_size=128,epochs=8,validation_split=0.2,verbose=1)

##hist = model.fit(train_data,labels,epochs=10)
model.save('request.h5')
##y_test = model.predict(encoded_data)
##y_pred = np.where(y_test > 0.5, 1, 0)
##print(y_pred[3])
##score = model.evaluate(encoded_data, labels, verbose=1)
##print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
