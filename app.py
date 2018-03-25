# -*- coding: utf-8 -*-

"""
Created on Mon Feb 26 18:48:35 2018

@author: Sundar Gsv
"""

# restore all of our data structures
import pickle
import tflearn
import tensorflow as tf
import random

data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('resources/intents.json') as json_data:
    intents = json.load(json_data)
    
####### reset underlying graph data #######
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard

# load our saved model
model = tflearn.DNN(net)
model.load('generatedModel/model.tflearn')

##### import helper functions #####
import helper

# create a data structure to hold user context
context = {}

def response(sentence, userID='123', show_details=False):
    results = helper.classify(sentence, model, words, classes)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return print(random.choice(i['responses']))

            results.pop(0)
  