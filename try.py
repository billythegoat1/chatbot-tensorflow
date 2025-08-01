#To know whether a file exist
import os

#To deal with json files(intents.json)
import json

#to deal with tokenization
import nltk

#To build our neural network
import torch

import torch.nn as nn

import torch.nn.functional as f

import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset



class ChatbotModel(nn.module):
    #Declare the constructor
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__

        self.fc1 = nn.linear(input_size, 128)
        self.fc2 = nn.linear(128, 64)
        self.fc3 = nn.linear(64, output_size)
        #To break linearity
        self.reLU = nn.ReLU()
        #To drop 50% of the output neurons to avoid overfitting
        self.dropout = nn.Dropout()

    def forward(self, x):
        #First layer
        x = self.reLU(self.fc1(x))
        x = self.dropout(x)

        #Second Layer
        x = self.reLU(self.fc2(x))
        x = self.dropout(x)

        #Third Layer
        x = self.reLU(self.fc3(x))
        x = self.dropout(x)


