import torch
import random
import numpy as np
from collections import deque
from main import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from plotResult import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

config = {
    'INPUT_SIZE': 12,
    'HIDDEN_SIZE': 256,
    'OUTPUT_SIZE': 3
}

class Agent:
    def __init__(self):
        self.numOfGames = 0
        self.epsilon = 0.0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # pop left most item
        self.model = Linear_QNet(config['INPUT_SIZE'], config['HIDDEN_SIZE'], config['OUTPUT_SIZE'])
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_d = Point(head.x, head.y - 20)
        point_u = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_d)),

            # Move direction
            dir_r,
            dir_d,
            dir_l,
            dir_u,

            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food down
            game.food.y > game.head.y, # food up

            # Distance from food
            abs(game.food.x - game.head.x) + abs(game.food.y - game.head.y)
        ]

        return np.array(state, dtype=int)
    def remember(self, state, action, reward, next_state, gameOver):
        self.memory.append((state, action, reward, next_state, gameOver))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, gameOvers = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, gameOvers)

    def train_short_memory(self, state, action, reward, next_state, gameOver):
        self.trainer.train_step(state, action, reward, next_state, gameOver)
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 90 - self.numOfGames
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        old_state = agent.get_state(game)

        # get move
        action = agent.get_action(old_state)

        # perform move and get new state
        reward, gameOver, score = game.play_step(action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(old_state, action, reward, state_new, gameOver)

        # remember
        agent.remember(old_state, action, reward, state_new, gameOver)

        if gameOver:
            # train long memory and plot result
            game.reset()
            agent.numOfGames += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.numOfGames, 'Score', score, 'Current Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.numOfGames
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
