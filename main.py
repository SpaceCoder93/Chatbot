from os import environ
import random
import json
import pickle
from nltk import translate
import numpy as np
from colorama import Fore
import os

from googletrans import Translator

import nltk
from nltk.stem import WordNetLemmatizer
from numpy.core.multiarray import fromiter, result_type

from tensorflow.keras.models import load_model

translator = Translator()

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(str(sentence))
    sentence_word = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def word_bag(sentence):
    sentence_word = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_word:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = word_bag(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
os.system('cls||clear')
print()
print(Fore.GREEN + 'Chatbot Booted Succesfully')

while True:
    print(Fore.CYAN)
    message = str(input(""))
    if message == "Quit":
        print(Fore.RED + 'Chatbot Shutting Down')
        break
    else:
        lang = translator.detect(message).lang
        message = translator.translate(message, dest='en')
        ints = predict_class(message)
        res = get_response(ints, intents)
        res = translator.translate(res, dest=lang)
        print(Fore.YELLOW + res.text)