import numpy as np 
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout


class Learner(object):
    def __init__(self):
        self.reward = 0
        self.alpha = 0.003
        self.model = self.buildModel()

    def buildModel(self):
        model = Sequential()
        model.add(Dense(units = 120, activation='relu', input_dim = 11))
        model.add(Dense(units = 3, activation='softmax'))
        opt = Adam(self.alpha)
        model.compile(loss='mse', optimizer=opt)

        return model

    def setReward(self, player, crash):
        self.reward = 0
        if crash:
            self.reward = -10
        if player.eaten:
            self.reward = 10
        return self.reward

    def getState(self):
        state = []

        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0

    def pushIntoMemory(self):
        pass

    def replay(self):
        pass

    def train(self):
        pass