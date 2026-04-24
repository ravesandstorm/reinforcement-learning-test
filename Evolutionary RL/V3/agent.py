import torch
import random
import numpy as np
import torch.multiprocessing as mp
from main import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import SnakeNet, EvolutionaryTrainer
from plotResult import plot

# Hyperparameters
POPULATION_SIZE = 100
GENERATIONS = 500
GAMES_PER_EVAL = 3
ELITISM_RATIO = 0.05
MATING_POOL_RATIO = 0.20

def get_state(game):
    """Calculates the 11-value state array for the NN based on current game view."""
    head = game.snake[0]
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)
    
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
        (dir_l and game.is_collision(point_u)) or 
        (dir_r and game.is_collision(point_d)),

        # Danger left
        (dir_d and game.is_collision(point_r)) or 
        (dir_u and game.is_collision(point_l)) or 
        (dir_r and game.is_collision(point_u)) or 
        (dir_l and game.is_collision(point_d)),
        
        # Move direction
        dir_l, dir_r, dir_u, dir_d,
        
        # Food location 
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y   # food down
    ]
    return np.array(state, dtype=int)

def worker_eval(args):
    """The function executed by each CPU core."""
    state_dict, seeds, render_flag = args
    
    # Rebuild model in local memory
    model = SnakeNet(11, 256, 3)
    model.load_state_dict(state_dict)
    model.eval()

    total_fitness = 0

    # Test across multiple seeded games
    for seed in seeds:
        game = SnakeGameAI(render=render_flag, seed=seed)
        game_over = False
        total_steps = 0
        
        while not game_over:
            state = get_state(game)
            state_tensor = torch.tensor(state, dtype=torch.float)
            
            with torch.no_grad():
                prediction = model(state_tensor)
            
            # Convert prediction to action [1,0,0], [0,1,0], or [0,0,1]
            move = [0, 0, 0]
            action_idx = int(torch.argmax(prediction).item())
            move[action_idx] = 1
            
            game_over, score, steps_survived = game.play_step(move)
            total_steps += 1
            
        # Fitness formula: High reward for food, tiny reward for surviving
        fitness = (score * 15) + (total_steps * 0.01)
        total_fitness += fitness

    return total_fitness / len(seeds)

def train():
    mp.set_start_method('spawn', force=True) # Required for PyTorch MP
    
    population = EvolutionaryTrainer.initialize_population(POPULATION_SIZE)
    plot_max_scores = []
    plot_mean_scores = []
    
    for generation in range(GENERATIONS):
        # 1. Generate Fixed Seeds for this generation
        current_seeds = [random.randint(0, 10000) for _ in range(GAMES_PER_EVAL)]
        
        # 2. Package arguments for multiprocessing
        # Only model 0 gets render=True
        worker_args = [(pop.state_dict(), current_seeds, i==0) for i, pop in enumerate(population)]
        
        # 3. Dispatch to CPU cores
        with mp.Pool(processes=mp.cpu_count()) as pool:
            fitness_scores = pool.map(worker_eval, worker_args)
            
        # 4. Analytics
        max_fitness = max(fitness_scores)
        mean_fitness = sum(fitness_scores) / POPULATION_SIZE
        
        plot_max_scores.append(max_fitness)
        plot_mean_scores.append(mean_fitness)
        plot(plot_max_scores, plot_mean_scores)
        
        print(f"Gen {generation} | Max Fit: {max_fitness:.2f} | Mean Fit: {mean_fitness:.2f}")

        # 5. Selection (Sort descending by fitness)
        scored_population = list(zip(fitness_scores, population))
        scored_population.sort(key=lambda x: x[0], reverse=True)
        sorted_population = [model for score, model in scored_population]
        
        # Save best model periodically
        if generation % 10 == 0:
            torch.save(sorted_population[0].state_dict(), f'best_snake_gen{generation}.pth')

        # 6. Elitism & Mating Pool Setup
        num_elites = int(POPULATION_SIZE * ELITISM_RATIO)
        num_mating = int(POPULATION_SIZE * MATING_POOL_RATIO)
        
        elites = sorted_population[:num_elites]
        mating_pool = sorted_population[:num_mating]
        
        # 7. Crossover & Mutation
        new_population = [np.copy.deepcopy(model) for model in elites]
        
        while len(new_population) < POPULATION_SIZE:
            # Tournament selection (randomly pick 2 from top 20%)
            p1 = random.choice(mating_pool)
            p2 = random.choice(mating_pool)
            
            child = EvolutionaryTrainer.crossover(p1, p2)
            EvolutionaryTrainer.mutate(child, mutation_rate=0.1, noise_std=0.05)
            new_population.append(child)
            
        population = new_population

if __name__ == '__main__':
    train()