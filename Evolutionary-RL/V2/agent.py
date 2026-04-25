import torch
import random
import numpy as np
import torch.multiprocessing as mp
from collections import deque
from main import SnakeGameNav, Direction, Point
from model import Linear_Net
from plotResult import plot
import copy

POPULATION_SIZE = 100
GAMES_PER_EVAL = 3
MUTATION_RATE = 0.1
MUTATION_NOISE = 0.05
ELITE_PERCENTAGE = 0.05
MATING_POOL_PERCENTAGE = 0.2

config = {
    'INPUT_SIZE': 11,
    'HIDDEN_SIZE': 256,
    'OUTPUT_SIZE': 3
}

def get_state(game):
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        (dir_r and game.is_collision(point_r)) or
        (dir_l and game.is_collision(point_l)) or
        (dir_u and game.is_collision(point_u)) or
        (dir_d and game.is_collision(point_d)),

        (dir_u and game.is_collision(point_r)) or
        (dir_d and game.is_collision(point_l)) or
        (dir_l and game.is_collision(point_u)) or
        (dir_r and game.is_collision(point_d)),

        (dir_d and game.is_collision(point_r)) or
        (dir_u and game.is_collision(point_l)) or
        (dir_r and game.is_collision(point_u)) or
        (dir_l and game.is_collision(point_d)),

        dir_l, dir_r, dir_u, dir_d,

        game.food.x < game.head.x,
        game.food.x > game.head.x,
        game.food.y < game.head.y,
        game.food.y > game.head.y,
    ]
    return np.array(state, dtype=int)

def evaluate_model(args):
    model_state_dict, seeds, render_flag = args
    model = Linear_Net(config['INPUT_SIZE'], config['HIDDEN_SIZE'], config['OUTPUT_SIZE'])
    model.load_state_dict(model_state_dict)
    model.eval()

    game = SnakeGameNav(render=render_flag)
    total_fitness = 0

    for seed in seeds:
        game.reset(seed)
        game_over = False
        fitness = 0
        while not game_over:
            state = get_state(game)
            state_tensor = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                prediction = model(state_tensor)
            move = int(torch.argmax(prediction).item())
            action = [0, 0, 0]
            action[move] = 1

            reward, game_over, score = game.play_step(action)
            fitness += reward
        total_fitness += fitness

    if render_flag and game.display:
        import pygame
        pygame.quit()

    return total_fitness / len(seeds)

def mutate_tensor(tensor):
    tensor_copy = tensor.clone()
    mutation_mask = torch.rand_like(tensor_copy) < MUTATION_RATE
    noise = torch.randn_like(tensor_copy) * MUTATION_NOISE
    return tensor_copy + (noise * mutation_mask)

def crossover(parent1_dict, parent2_dict):
    offspring_dict = {}
    for key in parent1_dict.keys():
        t1 = parent1_dict[key]
        t2 = parent2_dict[key]
        mask = torch.rand_like(t1) > 0.5
        offspring_dict[key] = torch.where(mask, t1, t2)
    return offspring_dict

def train():
    mp.set_start_method('spawn', force=True)

    population = [Linear_Net(config['INPUT_SIZE'], config['HIDDEN_SIZE'], config['OUTPUT_SIZE']) for _ in range(POPULATION_SIZE)]
    
    # Phase 1: Initialize with Gaussian noise
    for p in population:
        state_dict = p.state_dict()
        for key in state_dict:
            state_dict[key] += torch.randn_like(state_dict[key]) * 0.1
        p.load_state_dict(state_dict)

    plot_max_scores = []
    plot_mean_scores = []

    generation = 0

    while True:
        generation += 1

        # Generate seeds for the current generation
        seeds = [random.randint(0, 100000) for _ in range(GAMES_PER_EVAL)]

        # Prepare arguments for multiprocessing
        args_list = []
        for i, model in enumerate(population):
            render = (i == 0) # Only Render the first one
            args_list.append((model.state_dict(), seeds, render))

        print(f"Generation {generation} - Evaluating...")

        # Evaluate population
        with mp.Pool(processes=mp.cpu_count()) as pool:
            fitness_scores = pool.map(evaluate_model, args_list)

        # Pair models with their fitness
        population_fitness = list(zip(population, fitness_scores))
        population_fitness.sort(key=lambda x: x[1], reverse=True)

        max_fitness = population_fitness[0][1]
        mean_fitness = sum(fitness_scores) / len(fitness_scores)

        print(f"Gen {generation}: Max Fitness: {max_fitness:.2f}, Mean Fitness: {mean_fitness:.2f}")

        plot_max_scores.append(max_fitness)
        plot_mean_scores.append(mean_fitness)
        plot(plot_max_scores, plot_mean_scores)

        # Checkpointing
        if generation % 10 == 0:
            population_fitness[0][0].save(f'model_gen_{generation}.pth')

        # Reproduction (Elitism)
        num_elites = int(POPULATION_SIZE * ELITE_PERCENTAGE)
        elites = [copy.deepcopy(model) for model, _ in population_fitness[:num_elites]]

        # Mating Pool
        num_mating_pool = int(POPULATION_SIZE * MATING_POOL_PERCENTAGE)
        mating_pool = [model for model, _ in population_fitness[:num_mating_pool]]

        # Generate Offspring
        new_population = elites[:]
        num_offspring = POPULATION_SIZE - num_elites

        for _ in range(num_offspring):
            p1, p2 = random.sample(mating_pool, 2)
            
            # Crossover
            offspring_state_dict = crossover(p1.state_dict(), p2.state_dict())
            
            # Mutation
            for key in offspring_state_dict:
                offspring_state_dict[key] = mutate_tensor(offspring_state_dict[key])
                
            offspring = Linear_Net(config['INPUT_SIZE'], config['HIDDEN_SIZE'], config['OUTPUT_SIZE'])
            offspring.load_state_dict(offspring_state_dict)
            new_population.append(offspring)

        population = new_population

if __name__ == '__main__':
    train()
