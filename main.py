#to check if something exists
import os

#to work with json files
import json

#to pick a random intent if multiple are a choice
import random

#For tokenization
import nltk

#To deal with arrays and matrices
import numpy as np

#To build our neural network
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader, Tensordataset


#Building the infrastructure
class ChatbotModel(nn.Module):
    #Constructor for our infrastructure
    def __init__(self, input_size, output_size):
        #Calling the constructor from nn.Module
        super(ChatbotModel, self).__init__()

        #Defining the layers of our network(3 hidden layers)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

        #To break linearity
        self.relu = nn.ReLU()
        #Drop 50% of the neurons to avoid overfitting
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        #First Layer
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        #Second Layer
        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        #Third Layer
        x = self.fc3(x)

        #return the predicted intent
        return x

#This is the actual chatbot
class ChatbotAssistant:
    #Defining the constructor for our chatbot
    #The chatbot needs to know where the intents.json are(intents_path)
    #The chatbot needs a map of what function to call if this/that intent
    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.function_mappings = function_mappings

        #List of pattern-intent pairs to train input > intent label
        self.documents = []
        #List of all unique words for BoW vectors and model input
        self.vocabulary = []

        #the tags in the intents.json
        self.intents = []
        #The static responses still in intents.json
        self.intents_responses = []

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words]

        return words

    
    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]
    @staticmethod
    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer
        if os.path.exists(self.intense_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

    
            #Go through every intent dictionary
            for intent in intents_data['intents']:
                #if it's a new tag, append it to the dictionary
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

                self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = self.documents[0]
            bag = self.bag_of_words(words)

    




            
    


