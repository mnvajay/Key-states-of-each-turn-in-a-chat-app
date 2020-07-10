import tensorflow as tf
import numpy as np
from tensorflow import keras
import json
import codecs
import pickle
from keras.preprocessing.text import one_hot
from numpy import array

from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, GRU
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import MaxPool1D,Conv1D


# seed-fixing
from numpy.random import seed
seed(30)
tf.set_random_seed(30)

file_path = "../ontologies"
ontologies = json.load(codecs.open(file_path,"r","utf-8"))
nClasses = len(ontologies["informable"]["food"])

file_path = "inform-option1.json"
data = json.load(codecs.open(file_path,"r","utf-8"))

with open('../tokenization/tokenizer.pickle','rb') as handle:
   tokenizer = pickle.load(handle)

word_index = tokenizer.word_index

vocab = []
for idx in range(0,len(data)):
   vocab = vocab + data[idx]['dialog'].split()

vocab_size = len(set(vocab))

train_data = []
food_labels,area_labels,name_labels,price_labels = [],[],[],[]
n_food_labels = []

for idx in range(0,len(data)):
  train_data.append(data[idx]['dialog'])
  food_labels.append(data[idx]['food'])
  area_labels.append(data[idx]['area'])
##  name_labels.append(data[idx]['name'])
  price_labels.append(data[idx]['price range'])

food_labels = array(food_labels)
n_food_labels = len(set(j for i in food_labels for j in i))

area_labels = array(area_labels)
n_area_labels = len(set(j for i in area_labels for j in i))

##name_labels = array(name_labels)
##n_name_labels = len(set(j for i in name_labels for j in i))
##print(n_name_labels)

price_labels = array(price_labels)
n_price_labels = len(set(j for i in price_labels for j in i))


from sklearn.preprocessing import MultiLabelBinarizer
mlb_food = MultiLabelBinarizer()
mlb_food.fit([ontologies["informable"]["food"]])
food_labels = mlb_food.transform(food_labels)
food_labels = np.asarray(food_labels).astype(np.float32)

mlb_area = MultiLabelBinarizer()
mlb_area.fit([ontologies["informable"]["area"]])
area_labels = mlb_area.transform(area_labels)
area_labels = np.asarray(area_labels).astype(np.float32)

##mlb_name = MultiLabelBinarizer()
##name_labels = mlb_name.fit_transform(name_labels)
##name_labels = np.asarray(name_labels).astype(np.float32)

mlb_price = MultiLabelBinarizer()
mlb_price.fit([ontologies["informable"]["price range"]])
price_labels = mlb_price.transform(price_labels)
price_labels = np.asarray(price_labels).astype(np.float32)


##encoded_data = [one_hot(d,vocab_size) for d in train_data]
encoded_data = [[word_index[i] for i in sent.replace('?',' ').split()] for sent in train_data]

train_data = keras.preprocessing.sequence.pad_sequences(encoded_data,value=0,padding="post",maxlen=51,dtype=object)


import gensim
WORD2VEC_MODEL = "model_1_lac.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_MODEL,binary=True,limit=100000)

embedding_weights = np.zeros((2373,300))
for word,index in word_index.items():
    try:
        embedding_weights[index] = word2vec[word]
    except:
        pass


## food
model_food = Sequential()
embedding_layer = Embedding(2373,300,weights=[embedding_weights],input_length=51,trainable=False)
model_food.add(embedding_layer)
model_food.add(GRU(128))
model_food.add(Dense(len(ontologies["informable"]["food"]),activation='softmax'))
model_food.compile(optimizer="adam", loss="binary_crossentropy",metrics = ['accuracy'])
# print(model_food.summary())
food_hist = model_food.fit(train_data,food_labels,epochs=10,verbose=0)
model_food.save('food_inform.h5')

## area

model_area = Sequential()
embedding_layer = Embedding(2373,300,weights=[embedding_weights],input_length=51,trainable=False)
model_area.add(embedding_layer)
model_area.add(GRU(128))
model_area.add(Dense(len(ontologies["informable"]["area"]),activation='softmax'))
model_area.compile(optimizer="adam", loss="binary_crossentropy",metrics = ['accuracy'])
# print(model_area.summary())
area_hist = model_area.fit(train_data,area_labels,epochs=10,verbose=0)
model_area.save('area_inform.h5')

## price
model_price = Sequential()
embedding_layer = Embedding(2373,300,weights=[embedding_weights],input_length=51,trainable=False)
model_price.add(embedding_layer)
model_price.add(GRU(128))
model_price.add(Dense(len(ontologies["informable"]["price range"]),activation='softmax'))
model_price.compile(optimizer="adam", loss="binary_crossentropy",metrics = ['accuracy'])
# print(model_price.summary())
price_hist = model_price.fit(train_data,price_labels,epochs=10,verbose=0)
model_price.save('price_inform.h5')
# print(model_price.summary())

price_hist = model_price.fit(train_data,price_labels,epochs=10,verbose=0)
model_price.save('price_inform.h5')
