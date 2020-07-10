import json
import codecs
import keras
import pickle

from keras.preprocessing.text import Tokenizer

files = ["request-option1.json","request-option1-dev.json","request-option1-test.json"]

dialogues = []
vocab = []
for i in files:
    file_path = i
    data = json.load(codecs.open(file_path,"r","utf-8"))
    for idx in range(0,len(data)):
        dialogues.append(data[idx]['dialog'])
        vocab = vocab+data[idx]['dialog'].split()

MAX_NB_WORDS = len(set(vocab)) + 500
print(MAX_NB_WORDS)
tokenizer = Tokenizer(num_words=MAX_NB_WORDS,lower = True,char_level=False)
tokenizer.fit_on_texts(dialogues)
word_seq = tokenizer.texts_to_sequences(dialogues)
word_index = tokenizer.word_index

##with open('tokenizer.pickle','wb') as handle:
##    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)
print(word_index['town'])
