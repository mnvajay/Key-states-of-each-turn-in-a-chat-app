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
from keras.layers import Flatten, LSTM
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

train_data = keras.preprocessing.sequence.pad_sequences(encoded_data,value=0,padding="pre",maxlen=51,dtype=object)

## food
deep_inputs_food = Input(shape=(51,))
embedding_layer_food = Embedding(2373,64)(deep_inputs_food)
conv_1D_food = Conv1D(filters=64,kernel_size=3,activation="relu")(embedding_layer_food)
drop_out_food = Dropout(0.2)(conv_1D_food)
max_pool_food = GlobalMaxPooling1D()(drop_out_food)
dense_layer_food = Dense(len(ontologies["informable"]["food"]),activation='softmax')(max_pool_food)
model_food = Model(inputs=[deep_inputs_food],outputs=dense_layer_food)

model_food.compile(optimizer="adam", loss="binary_crossentropy",metrics = ['accuracy'])
model_food.summary()
food_hist = model_food.fit(train_data,food_labels,epochs=10,verbose=0)
model_food.save('food_inform.h5')

##y_test = model_food.predict(train_data)
##print(np.where(y_test[3] == max(y_test[3])))
##print(labels[3])


## area

deep_inputs_area = Input(shape=(51,))
embedding_layer_area = Embedding(2373,64)(deep_inputs_area)
conv_1D_area = Conv1D(filters=64,kernel_size=3,activation="relu")(embedding_layer_area)
drop_out_area = Dropout(0.2)(conv_1D_area)
max_pool_area = GlobalMaxPooling1D()(drop_out_area)
dense_layer_area = Dense(len(ontologies["informable"]["area"]),activation='softmax')(max_pool_area)
model_area = Model(inputs=[deep_inputs_area],outputs=dense_layer_area)

model_area.compile(optimizer="adam", loss="binary_crossentropy",metrics = ['accuracy'])
model_area.summary()

area_hist = model_area.fit(train_data,area_labels,epochs=10,verbose=0)
model_area.save('area_inform.h5')


#### name
##
##deep_inputs_name = Input(shape=(51,))
##embedding_layer_name = Embedding(vocab_size+1,64)(deep_inputs_name)
##conv_1D_name = Conv1D(filters=64,kernel_size=3,activation="relu")(embedding_layer_name)
##drop_out_name = Dropout(0.2)(conv_1D_name)
##max_pool_name = GlobalMaxPooling1D()(drop_out_name)
##dense_layer_name = Dense(n_name_labels,activation='softmax')(max_pool_name)
##model_name = Model(inputs=[deep_inputs_name],outputs=dense_layer_name)
##
##model_name.compile(optimizer="adam", loss="binary_crossentropy",metrics = ['accuracy'])
##model_name.summary()
##
##name_hist = model_name.fit(train_data,name_labels,epochs=10)

## price

deep_inputs_price = Input(shape=(51,))
embedding_layer_price = Embedding(2373,64)(deep_inputs_price)
conv_1D_price = Conv1D(filters=64,kernel_size=3,activation="relu")(embedding_layer_price)
drop_out_price = Dropout(0.2)(conv_1D_price)
max_pool_price = GlobalMaxPooling1D()(drop_out_price)
dense_layer_price = Dense(len(ontologies["informable"]["price range"]),activation='softmax')(max_pool_price)
model_price = Model(inputs=[deep_inputs_price],outputs=dense_layer_price)

model_price.compile(optimizer="adam", loss="binary_crossentropy",metrics = ['accuracy'])
model_price.summary()
price_hist = model_price.fit(train_data,price_labels,epochs=10,verbose=0)
model_price.save('price_inform.h5')

