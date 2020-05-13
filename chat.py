import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tensorflow
import tflearn
import json
import pickle
import random

try:
    with open("chatbot_data.json") as file:
        data = json.load(file)
except:
    print("chatbot_data.json is not found")

try:
    with open("data.pickle","rb") as f:
        words,labels,training,output = pickle.load(f)
except:
    print("data.pickel file not found\nMake sure you run start_or_update.py first")

tensorflow.reset_default_graph()

net =tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    print("Trained data not found\nMake sure training is done with train.py")
else:
    def bag_of_words(s,words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i,w in enumerate(words):
                if w == se:
                    bag[i] = 1
        return numpy.array(bag)

    def chat():
        print("Start talking")
        print("(type 'quit' to exit)")
        while True:
            inp = input("YOU: ")
            if inp.lower() == "quit":
                break

            results = model.predict([bag_of_words(inp,words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]

            if results[results_index] > 0.7:
                for tg in data["intents"]:
                    if tg["tag"]==tag:
                        responses = tg["responses"]

                print(random.choice(responses))
            else:
                print("Sorry I don't know that")

    chat()