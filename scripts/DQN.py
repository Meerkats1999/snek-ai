import numpy as np 
import random
import pandas as pd
from operator import add
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout

class Learner(object):
    def __init__(self):
        self.reward = 0
        self.alpha = 0.003
        self.gamma = 1
        self.memory = []
        self.rndNum = 0
        self.model = self.buildModel()
        self.df = pd.DataFrame()

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

    def fetchState(self, game, player, food):
        state = [
            ((player.x_change == 20 and player.y_change == 0 and 
            ((list(map(add, player.position[-1], [20, 0])) in player.position) or
            player.position[-1][0] + 20 >= (game.gameWidth - 20))) or 
            
            (player.x_change == -20 and player.y_change == 0 and 
            ((list(map(add, player.position[-1], [-20, 0])) in player.position) or
            player.position[-1][0] - 20 < 20)) or 
            
            (player.x_change == 0 and player.y_change == -20 and 
            ((list(map(add, player.position[-1], [0, -20])) in player.position) or
            player.position[-1][-1] - 20 < 20)) or 
            
            (player.x_change == 0 and player.y_change == 20 and 
            ((list(map(add, player.position[-1], [0, 20])) in player.position) or
            player.position[-1][-1] + 20 >= (game.gameHeight-20)))), 

            ((player.x_change == 0 and player.y_change == -20 and 
            ((list(map(add,player.position[-1],[20, 0])) in player.position) or 
            player.position[ -1][0] + 20 > (game.gameWidth-20))) or 
            
            (player.x_change == 0 and player.y_change == 20 and 
            ((list(map(add,player.position[-1],[-20,0])) in player.position) or 
            player.position[-1][0] - 20 < 20)) or 
            
            (player.x_change == -20 and player.y_change == 0 and 
            ((list(map(add,player.position[-1],[0,-20])) in player.position) or 
            player.position[-1][-1] - 20 < 20)) or 
            
            (player.x_change == 20 and player.y_change == 0 and 
            ((list(map(add,player.position[-1],[0,20])) in player.position) or 
            player.position[-1][-1] + 20 >= (game.gameHeight-20)))),  

            ((player.x_change == 0 and player.y_change == 20 and 
            ((list(map(add,player.position[-1],[20,0])) in player.position) or
            player.position[-1][0] + 20 > (game.gameWidth-20))) or 
            
            (player.x_change == 0 and player.y_change == -20 and 
            ((list(map(add, player.position[-1],[-20,0])) in player.position) or 
            player.position[-1][0] - 20 < 20)) or 
            
            (player.x_change == 20 and player.y_change == 0 and 
            ((list(map(add,player.position[-1],[0,-20])) in player.position) or 
            player.position[-1][-1] - 20 < 20)) or 
            
            (player.x_change == -20 and player.y_change == 0 and 
            ((list(map(add,player.position[-1],[0,20])) in player.position) or
            player.position[-1][-1] + 20 >= (game.gameHeight-20)))),

            player.x_change == -20,  
            player.x_change == 20,  
            player.y_change == -20,  
            player.y_change == 20,  
            food.x_food < player.x, 
            food.x_food > player.x,  
            food.y_food < player.y,  
            food.y_food > player.y  
        ]

        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0

        return np.asarray(state)

    def pushIntoMemory(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def replayTrain(self, memory):
        if len(memory) > 1000:
            batch = random.sample(memory, 1000)
        else:
            batch = memory
        for state, action, reward, nextState, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([nextState]))[0])
            finalTarget = self.model.predict(np.array([state]))
            finalTarget[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), finalTarget, epochs=1, verbose=0)

    def shortTrain(self, state, action, reward, nextState, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(nextState.reshape((1,11)))[0])
        finalTarget = self.model.predict(state.reshape((1,11)))
        finalTarget[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1,11)), finalTarget, epochs=1, verbose=0)