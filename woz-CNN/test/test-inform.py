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

file_path = "../ontologies"
ontologies = json.load(codecs.open(file_path,"r","utf-8"))

word_index = tokenizer.word_index

file_path = "../training/inform-option1.json"
data = json.load(codecs.open(file_path,"r","utf-8"))


test_data = []
test_food_labels = []
test_area_labels = []
test_price_labels = []

for idx in range(0,len(data)):
    test_data.append(data[idx]['dialog'])
    test_food_labels.append(data[idx]['food'])
    test_area_labels.append(data[idx]['area'])
    test_price_labels.append(data[idx]['price range'])

from sklearn.preprocessing import MultiLabelBinarizer
mlb_food = MultiLabelBinarizer()
mlb_food.fit([ontologies['informable']['food']])
food_labels = array(mlb_food.transform(test_food_labels))

mlb_area = MultiLabelBinarizer()
mlb_area.fit([ontologies['informable']['area']])
area_labels = array(mlb_area.transform(test_area_labels))

mlb_price = MultiLabelBinarizer()
mlb_price.fit([ontologies['informable']['price range']])
price_labels = array(mlb_price.transform(test_price_labels))

encoded_data = [[word_index[i] for i in sent.replace('?',' ').split()] for sent in test_data]
test_data = keras.preprocessing.sequence.pad_sequences(encoded_data,value=0,padding="pre",maxlen=51,dtype=object)

model_food = keras.models.load_model("../training/food_inform.h5")
food_pred = model_food.predict(test_data)
print(test_data.shape)
food_pred = np.where(food_pred>0.5,1,0)
print(accuracy_score(food_pred,food_labels))

##pred = np.array([])
##for i in food_pred:
####    print(i)
##    k = np.where(i==np.max(i),1,0)
##    pred = np.append(pred,k)
##pred = pred.astype(np.int).reshape(len(test_data),len(ontologies['informable']['food']))
##print(pred)

model_area = keras.models.load_model("../training/area_inform.h5")
area_pred = model_area.predict(test_data)
area_pred = np.where(area_pred>0.5,1,0)
print(accuracy_score(area_pred,area_labels))

model_price = keras.models.load_model("../training/price_inform.h5")
price_pred = model_price.predict(test_data)
price_pred = np.where(price_pred>0.5,1,0)
print(accuracy_score(price_pred,price_labels))

##sample = np.array([[0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
## 0, 0, 0, 0, 0, 0, 0, 0, 18, 13, 48, 383, 26, 5, 16]])
##print(sample.shape)
##p = model_food.predict(sample)
##pred = np.where(p == np.amax(p), 1, 0)
##pred = mlb_food.inverse_transform(pred)
##print(pred)
