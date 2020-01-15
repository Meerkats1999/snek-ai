import pygame
from random import randint
import numpy as np
from keras.utils import to_categorical
from scripts.DQN import Learner

displayOption = True
speed = 0
gameCount = 100
pygame.font.init()


class Game:

    def __init__(self, gameWidth, gameHeight):
        self.gameWidth = gameWidth
        self.gameHeight = gameHeight
        self.gameDisplay = pygame.display.set_mode((gameWidth, gameHeight+60))
        self.bg = pygame.image.load("assets/images/background.jpg")
        self.crash = False
        self.player = Player(self)
        self.food = Food()
        self.score = 0


class Player(object):

    def __init__(self, game):
        x = 0.45 * game.gameWidth
        y = 0.5 * game.gameHeight
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False
        self.image = pygame.image.load('assets/images/snek.jpeg')
        self.x_change = 20
        self.y_change = 0

    def updatePosition(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    def move(self):
        keys = pygame.key.get_pressed()
        for key in keys:
            if keys[pygame.K_LEFT]:
                move  = np.array([0 ,0, 1])

            elif keys[pygame.K_RIGHT]:
                move = np.array([0, 1, 0])

            elif keys[pygame.K_UP]:
                move = np.array([0, 0, 1])

            elif keys[pygame.K_DOWN]:
                move  = np.array([0, 1, 0])

            else:
                move = np.array([1, 0, 0])

        return move

    def doMove(self, move, x, y, game, food, dqn):
        move_array = [self.x_change, self.y_change]

        if self.eaten:

            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1
        if np.array_equal(move ,[1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(move,[0, 1, 0]) and self.y_change == 0:  
            move_array = [0, self.x_change]
        elif np.array_equal(move,[0, 1, 0]) and self.x_change == 0:  
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  
            move_array = [0, -self.x_change]
        elif np.array_equal(move,[0, 0, 1]) and self.x_change == 0: 
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        if self.x < 20 or self.x > game.gameWidth-40 or self.y < 20 or self.y > game.gameHeight-40 or [self.x, self.y] in self.position:
            game.crash = True
        eat(self, food, game)

        self.updatePosition(self.x, self.y)

    def displayPlayer(self, x, y, food, game):
        self.position[-1][0] = x
        self.position[-1][1] = y

        if game.crash == False:
            for i in range(food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))

            updateScreen()
        else:
            pygame.time.wait(300)


class Food(object):

    def __init__(self):
        self.x_food = 240
        self.y_food = 200
        self.image = pygame.image.load('assets/images/fruit.png')

    def foodPosition(self, game, player):
        x_rand = randint(20, game.gameWidth - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.gameHeight - 40)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position:
            return self.x_food, self.y_food
        else:
            self.foodPosition(game,player)

    def displayFood(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        updateScreen()


def eat(player, food, game):
    if player.x == food.x_food and player.y == food.y_food:
        food.foodPosition(game, player)
        player.eaten = True
        game.score = game.score + 1


def fetchRecord(score, record):
        if score >= record:
            return score
        else:
            return record


def displayUI(game, score, record, gameCounter):
    font = pygame.font.Font('assets/fonts/Roboto/Roboto-Italic.ttf', 24)
    fontBold = pygame.font.Font('assets/fonts/Roboto/Roboto-Bold.ttf', 22)
    gameNum = font.render('GAME: ', True, (0, 0, 0))
    gameNumNumber = fontBold.render(str(gameCounter + 1), True, (0, 0, 0))
    textScore = font.render('SCORE: ', True, (0, 0, 0))
    textScoreNumber = fontBold.render(str(score), True, (0, 0, 0))
    text_highest = font.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = fontBold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(gameNum, (40, 660))
    game.gameDisplay.blit(gameNumNumber, (140, 660))
    game.gameDisplay.blit(textScore, (200, 660))
    game.gameDisplay.blit(textScoreNumber, (300, 660))
    game.gameDisplay.blit(text_highest, (360, 660))
    game.gameDisplay.blit(text_highest_number, (560, 660))
    game.gameDisplay.blit(game.bg, (10, 10))


def display(player, food, game, record, gameCounter):
    game.gameDisplay.fill((255, 255, 255))
    displayUI(game, game.score, record, gameCounter)
    player.displayPlayer(player.position[-1][0], player.position[-1][1], player.food, game)
    food.displayFood(food.x_food, food.y_food, game)


def updateScreen():
    pygame.display.update()


def initializeGame(player, game, food, dqn):
    state_init1 = dqn.fetchState(game, player, food)
    action = [1, 0, 0]
    player.doMove(action, player.x, player.y, game, food, dqn)
    state_init2 = dqn.fetchState(game, player, food)
    reward1 = dqn.setReward(player, game.crash)
    dqn.pushIntoMemory(state_init1, action, reward1, state_init2, game.crash)
    dqn.replayTrain(dqn.memory)
    

# def run():
#     game = Game(640, 640)
#     player1 = game.player
#     food1 = game.food
#     record = 0

#     initializeGame(player1, game, food1)
#     if displayOption:
#         display(player1, food1, game, record)

#     while not game.crash:
#         pygame.time.delay(speed)

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()

#             move = player1.move()
            
#         player1.doMove(move, player1.x, player1.y, game, food1)
                 
#         display(player1,food1,game,record)
    
#     print('Score: ', game.score)


def trainRun():
    game = Game(640,640)
    player1 = game.player
    food1 = game.food
    record = 0
    gameCounter = 0
    dqn = Learner()
    
    while gameCounter < gameCount:
        game = Game(640,640)
        player1 = game.player
        food1 = game.food

        initializeGame(player1, game, food1, dqn)
        if displayOption:
            display(player1, food1, game, record, gameCounter)

        while not game.crash:
            dqn.rndNum = gameCount/2 - gameCounter

            oldState = dqn.fetchState(game, player1, food1)

            if randint(0, 100) < dqn.rndNum:
                finalMove = to_categorical(randint(0,2), num_classes=3)
            else:
                prediction = dqn.model.predict(oldState.reshape((1,11)))
                finalMove = to_categorical(np.argmax(prediction[0]), num_classes=3)

            player1.doMove(finalMove, player1.x, player1.y, game, food1, dqn)
            newState = dqn.fetchState(game, player1, food1)

            reward = dqn.setReward(player1, game.crash)

            dqn.shortTrain(oldState, finalMove, reward, newState, game.crash)

            dqn.pushIntoMemory(oldState, finalMove, reward, newState, game.crash)
            record = fetchRecord(game.score, record)

            if displayOption:
                display(player1, food1, game, record, gameCounter)
                pygame.time.wait(speed)

        dqn.replayTrain(dqn.memory)
        gameCounter += 1

        print('Game: ',gameCounter,'Score: ',game.score)

    pygame.quit()


if __name__ == "__main__":
    pygame.init()
    trainRun()