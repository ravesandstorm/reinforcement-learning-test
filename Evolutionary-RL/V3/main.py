import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font(pygame.font.get_default_font(), 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')
BLOCK_SIZE = 20
SPEED = 40 # Adjust for visual spectator speed

class SnakeGameAI:
    def __init__(self, w=640, h=480, render=False, seed=None):
        self.w = w
        self.h = h
        self.render_mode = render
        
        if self.render_mode:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake ERL Spectator')
            self.clock = pygame.time.Clock()
            
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                        Point(self.head.x-BLOCK_SIZE, self.head.y),
                        Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

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
            x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
            y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            candidate = Point(x, y)
            if candidate not in self.snake:
                self.food = candidate
                return

    def play_step(self, action):
        self.frame_iteration += 1

        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        self._move(action)
        self.snake.insert(0, self.head)

        game_over = False
        empty_space = (self.w // BLOCK_SIZE) * (self.h // BLOCK_SIZE) - len(self.snake)

        # starvation_limit = 100 * len(self.snake) # Basic starvation limit based on snake length
        # starvation_limit = self.max_steps_without_food + 1.5 * len(self.snake)  # dynamic starvation threshold based on snake length
        # starvation_limit = empty_space  # starvation threshold based on remaining empty space
        starvation_limit = len(self.snake) + 0.5 * empty_space  # dynamic starvation threshold based on snake length and empty space
        
        if self.is_collision() or self.frame_iteration > starvation_limit:
            game_over = True
            return game_over, self.score, self.frame_iteration

        if self.head == self.food:
            self.score += 1
            self.frame_iteration = 0 # Reset starvation timer
            self._place_food()
        else:
            self.snake.pop()

        if self.render_mode:
            self._update_ui()
            self.clock.tick(SPEED)
            
        return game_over, self.score, self.frame_iteration

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn

        self.direction = new_dir
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction == Direction.UP: y -= BLOCK_SIZE
        self.head = Point(x, y)

    def _update_ui(self):
        self.display.fill((0,0,0))
        for pt in self.snake:
            pygame.draw.rect(self.display, (0,255,0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        if self.food is not None:
            pygame.draw.rect(self.display, (255,0,0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, (255,255,255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()