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

##import models
request_model = keras.models.load_model("../training/request.h5")
food_model = keras.models.load_model("../training/food_inform.h5")
area_model = keras.models.load_model("../training/area_inform.h5")
price_model = keras.models.load_model("../training/price_inform.h5")

file_path = "../ontologies"
ontologies = json.load(codecs.open(file_path,"r","utf-8"))

last_question = '' 
turn_label = {
    'request':[],
    'food':'',
    'price':'',
    'area':''
    }
prev = {
    'user_utterance':'',
    'system_utterance':''
    }

## Tokenizer

with open('../tokenization/tokenizer.pickle','rb') as handle:
   tokenizer = pickle.load(handle)
word_index = tokenizer.word_index

def process_turn_hyp(transcription):
    """
    Returns the clean (i.e. handling interpunction signs) string for the given language. 
    """
    exclude = set(string.punctuation)
    exclude.remove("'")
    exclude.remove("?")

    transcription = ''.join(ch for ch in transcription if ch not in exclude)

    transcription = transcription.lower()
    transcription = transcription.replace(u"’", "'")
    transcription = transcription.replace(u"‘", "'")
    transcription = transcription.replace("don't", "dont")
    transcription = transcription.replace("'", "")
    
    return transcription

def embedding(transcript):
    encoded_data = [word_index[i] for i in transcript.replace('?',' ').split()]
    return encoded_data

def request_predict(transcript):
    pred = request_model.predict([transcript])
    request_mlb = MultiLabelBinarizer()
    request_mlb.fit([['address','area' ,'food', 'name', 'phone', 'postcode', 'price range']])
    pred = np.where(pred > 0.5, 1, 0)
    pred = request_mlb.inverse_transform(pred)
    if len(pred)>0:
        turn_label['request'] = [i for i in pred]

def inform_predict(transcript):
    food_mlb = MultiLabelBinarizer()
    food_mlb.fit([ontologies['informable']['food']])
    food_pred = food_model.predict([transcript])
    pred_food = np.where(food_pred==np.amax(food_pred),1,0)
    if len(pred_food)>0 and np.amax(food_pred)>0.1:
        pred_food = food_mlb.inverse_transform(pred_food)
##        print('food')
##        print(pred_food)
        turn_label['food'] = pred_food[0][0]

    area_mlb = MultiLabelBinarizer()
    area_mlb.fit([ontologies['informable']['area']])
    area_pred = area_model.predict([transcript])
    pred_area = np.where(area_pred==np.amax(area_pred),1,0)
    if len(pred_area)>0 and np.amax(area_pred)>0.4:
        pred_area = area_mlb.inverse_transform(pred_area)
##        print('area')
##        print(pred_area)
        turn_label['area'] = pred_area[0][0]

    price_mlb = MultiLabelBinarizer()
    price_mlb.fit([ontologies['informable']['price range']])
    price_pred = price_model.predict([transcript])
    pred_price = np.where(price_pred==np.amax(price_pred),1,0)
    if len(pred_price)>0 and np.amax(price_pred)>0.5:
        pred_price = price_mlb.inverse_transform(pred_price)
##        print('price')
##        print(pred_price)
        turn_label['price'] = pred_price[0][0]
    return ''


def mhello():
    mtext = ment.get()
##    print(mtext)
    flag = False
    user_input, system_input = '', ''
##    user and system utterance    
    if ('?' in system_input):
        last_question = process_turn_hyp(mtext)[1:]
    if ('U:' in mtext):
        user_input = process_turn_hyp(mtext)[1:]
        system_input = prev['system_utterance']
        flag = True
    else:
        system_input = process_turn_hyp(mtext)[1:]
        user_input = prev['user_utterance']
    sent = ''
    
    if (flag and user_input[:2]=='no'):
        sent = last_question
    else:
        sent = user_input + " " + system_input

    encoded_utter = embedding(sent)
    utterance = keras.preprocessing.sequence.pad_sequences([encoded_utter],value=0,padding="pre",maxlen=51,dtype=object)
##    print(utterance)
    request = request_predict(utterance)
    inform = inform_predict(utterance)

##    print(request)
    if (flag):
        mlabel2 = Label(mGui,text=mtext,anchor='ne').pack(fill='both')
##        x = []
##        for key in turn_label.keys():
##            x.append(key)
    else:
        mlabel3 = Label(mGui,text=mtext,anchor='sw').pack(fill='both')
##        x = []
##        for key in turn_label.keys():
##            x.append(key)
    print(turn_label)
    flag = False
    prev['user_utterance'] = user_input
    prev['system_utterance'] = system_input
    return

mGui = Tk()
ment = StringVar()
mGui.geometry('500x500+500+300')

mlabel = Label(mGui,text='My label').pack()
mbutton = Button(mGui,text='OK',command=mhello,fg='red',bg='blue').pack()
##mbutton.grid(row=10,column=0)
mEntry = Entry(mGui,textvariable=ment).pack()
mGui.mainloop()
