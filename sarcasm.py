# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 08:52:58 2019

@author: fkiaie
"""
import wget
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

wget.download('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json')

import json
with open('sarcasm.json', 'r') as f:
    datastore=json.load(f)
    
sentences = []
labels = []
urls =[]

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
    

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

print(len(word_index))
print(word_index)
sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, padding='post')
print(sentences[2])
print(padded[2])
print(padded.shape)

