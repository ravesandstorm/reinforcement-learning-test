import random
from collections import namedtuple
from enum import Enum
from typing import Optional

import pygame

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple("Point", "x y")

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20

class SnakeGameAI:
    """
    Snake environment that can run in two modes:
        - render=True: full pygame rendering + event loop pumping
        - render=False: headless logic-only simulation (fast rollouts)

    Reward shaping (used by the evolutionary fitness pipeline):
        +15 for food
        +0.001 per step survived
        -10 for collision or starvation
        starvation cutoff via max_steps_without_food
    """

    def __init__(
        self,
        w: int = 640,
        h: int = 480,
        render: bool = False,
        seed: Optional[int] = None,
        speed: int = 120,
        max_steps_without_food: int = 120,
    ):
        self.w = w
        self.h = h
        self.render = render
        self.speed = speed
        self.max_steps_without_food = max_steps_without_food

        self._rng = random.Random(seed)

        self.display = None
        self.clock = pygame.time.Clock()
        self.font = None

        if self.render:
            pygame.init()
            self.font = pygame.font.SysFont("arial", 25)
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake AI - Evolutionary RL")
            self.clock = pygame.time.Clock()

        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = random.Random(seed)

        self.direction = Direction.RIGHT

        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        self.score = 0
        self.frame_iteration = 0
        self.steps_since_food = 0
        self.food = Point(0, 0)
        self._place_food()

    def _place_food(self):
        all_points = [Point(x, y) for x in range(0, self.w, BLOCK_SIZE) 
                for y in range(0, self.h, BLOCK_SIZE)]
    
        # If the snake is covering more than 80% of the board, switch to mapping
        if len(self.snake) > (len(all_points) * 0.8):
            empty_spaces = [p for p in all_points if p not in self.snake]
            if empty_spaces:
                self.food = self._rng.choice(empty_spaces)
                return
        
        while True:
            x = self._rng.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = self._rng.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            candidate = Point(x, y)
            if candidate not in self.snake:
                self.food = candidate
                return

    def is_collision(self, pt: Optional[Point] = None) -> bool:
        if pt is None:
            pt = self.head

        if pt.x > self.w - BLOCK_SIZE or pt.x < 0:
            return True
        if pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True

        return False

    def play_step(self, action):
        self.frame_iteration += 1
        self.steps_since_food += 1

        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0.001  # survival reward per step
        game_over = False

        empty_space = (self.w // BLOCK_SIZE) * (self.h // BLOCK_SIZE) - len(self.snake)
        # starvation = self.steps_since_food >= self.max_steps_without_food
        # starvation = self.steps_since_food >= self.max_steps_without_food + 1.5 * len(self.snake)  # dynamic starvation threshold based on snake length
        # starvation = self.steps_since_food >= empty_space  # starvation threshold based on remaining empty space
        starvation = self.steps_since_food >= len(self.snake) + 0.5 * empty_space  # dynamic starvation threshold based on snake length and empty space
        if self.is_collision() or starvation:
            game_over = True
            reward -= 10.0
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward += 15.0
            self.steps_since_food = 0
            self._place_food()
        else:
            self.snake.pop()

        if self.render:
            self._update_ui()
            self.clock.tick(self.speed)

        return reward, game_over, self.score

    def _update_ui(self):
        if self.display is None or self.font is None:
            return

        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # action expected: [straight, right_turn, left_turn]
        action = tuple(int(v) for v in action)

        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if action == (1, 0, 0):
            new_dir = clockwise[idx]
        elif action == (0, 1, 0):
            new_dir = clockwise[(idx + 1) % 4]
        else:  # (0, 0, 1)
            new_dir = clockwise[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        else:
            y += BLOCK_SIZE

        self.head = Point(x, y)

    def close(self):
        if self.render:
            pygame.display.quit()
            pygame.quit()
