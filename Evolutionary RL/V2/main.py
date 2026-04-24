import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 300  # Faster for evolutionary testing

class SnakeGameNav:
    def __init__(self, w=640, h=480, render=False):
        self.w = w
        self.h = h
        self.render_flag = render
        if self.render_flag:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Evolutionary Snake')
        else:
            self.display = None
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frameIteration = 0
        self.steps_survived = 0
        self.steps_without_food = 0
        
    def _place_food(self):
        all_points = [Point(x, y) for x in range(0, self.w, BLOCK_SIZE) 
                for y in range(0, self.h, BLOCK_SIZE)]

        # If the snake is covering more than 80% of the board, switch to mapping
        if len(self.snake) > (len(all_points) * 0.8):
            empty_spaces = [p for p in all_points if p not in self.snake]
            if empty_spaces:
                self.food = random.choice(empty_spaces)
                return

        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            candidate = Point(x, y)
            if candidate not in self.snake:
                self.food = candidate
                return
            
    def play_step(self, action):
        self.frameIteration += 1
        self.steps_survived += 1
        self.steps_without_food += 1
        
        if self.render_flag:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0.001  # +0.001 per step survived
        game_over = False
        empty_space = (self.w // BLOCK_SIZE) * (self.h // BLOCK_SIZE) - len(self.snake)

        # starvation_limit = 100 * len(self.snake)  # Basic starvation limit based on snake length
        # starvation_limit = self.max_steps_without_food + 1.5 * len(self.snake)  # dynamic starvation threshold based on snake length
        # starvation_limit = empty_space  # starvation threshold based on remaining empty space
        starvation_limit = len(self.snake) + 0.5 * empty_space  # dynamic starvation threshold based on snake length and empty space

        if self.is_collision() or self.steps_without_food > starvation_limit:
            game_over = True
            if self.is_collision():
                reward -= 10
            elif self.steps_without_food > 100 * len(self.snake):
                reward -= 5
            return reward, game_over, self.score
            
        if self.head == self.food:
            self.score += 1
            reward = 15  # +15 for food
            self.steps_without_food = 0
            self._place_food()
        else:
            self.snake.pop()
            
        if self.render_flag:
            self._update_ui()
            self.clock.tick(SPEED)
            
        return reward, game_over, self.score
        
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0:
            return True
        if pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False
        
    def _update_ui(self):
        if self.display is None:
            return
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        if self.food is not None:
            pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        self.head = Point(x, y)
