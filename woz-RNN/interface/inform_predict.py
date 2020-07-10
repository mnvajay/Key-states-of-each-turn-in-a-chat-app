import sys
from tkinter import *
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MultiLabelBinarizer
import json
import codecs
import string
import pickle
import numpy as np

food_model = keras.models.load_model("../training/food_inform.h5")
area_model = keras.models.load_model("../training/area_inform.h5")
price_model = keras.models.load_model("../training/price_inform.h5")

file_path = "../ontologies"
ontologies = json.load(codecs.open(file_path,"r","utf-8"))
print('ahay')

def predict(transcript):
    print('a')
    food_mlb = MultiLabelBinarizer()
    food_mlb.fit([ontologies['informable']['food']])
    food_pre = food_model.predict([transcript])
    food_pred = np.where(food_pred==np.amax(food_pred),1,0)
    food_pred = food_mlb.inverse_transform(food_pred)
    print(food_pred)

    print(food_pred[0][0])


    area_mlb = MultiLabelBinarizer()
    area_mlb.fit([ontologies['informable']['area']])
    area_pre = area_model.predict([transcript])
    area_pred = np.where(area_pred==np.amax(area_pred),1,0)
    area_pred = food_mlb.inverse_transform(area_pred)

    print(area_pred[0][0])

    price_mlb = MultiLabelBinarizer()
    price_mlb.fit([ontologies['informable']['price range']])
    price_pre = price_model.predict([transcript])
    price_pred = np.where(price_pred==np.amax(price_pred),1,0)
    price_pred = price_mlb.inverse_transform(price_pred)

    print(price_pred[0][0])

    return {'food':food_pred[0][0],'area':area_pred[0][0],'price':price_pred[0][0]}
