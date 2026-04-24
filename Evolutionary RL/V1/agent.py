import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

from main import Direction, Point, SnakeGameAI
from model import LinearQNet, add_gaussian_noise_, clone_state_dict, mutate_state_dict, uniform_crossover
from plotResult import EvolutionPlotter

@dataclass
class EvolutionConfig:
    input_size: int = 11
    hidden_size: int = 256
    output_size: int = 3

    population_size: int = 100
    generations: int = 200
    games_per_model: int = 3

    elite_fraction: float = 0.05
    mating_fraction: float = 0.20

    mutation_rate: float = 0.10
    mutation_std: float = 0.02
    initial_noise_std: float = 0.02

    max_steps_without_food: int = 120

    checkpoint_every: int = 10
    checkpoint_dir: str = "checkpoints"

    num_workers: int = max(1, (os.cpu_count() or 2) - 1)
    render_first_worker: bool = True
    render_speed: int = 90

    seed: int = 42

def get_state(game: SnakeGameAI) -> np.ndarray:
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
        # danger straight
        (dir_r and game.is_collision(point_r))
        or (dir_l and game.is_collision(point_l))
        or (dir_u and game.is_collision(point_u))
        or (dir_d and game.is_collision(point_d)),

        # danger right
        (dir_u and game.is_collision(point_r))
        or (dir_d and game.is_collision(point_l))
        or (dir_l and game.is_collision(point_u))
        or (dir_r and game.is_collision(point_d)),

        # danger left
        (dir_d and game.is_collision(point_r))
        or (dir_u and game.is_collision(point_l))
        or (dir_r and game.is_collision(point_u))
        or (dir_l and game.is_collision(point_d)),

        # move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,

        # food location
        game.food.x < game.head.x,
        game.food.x > game.head.x,
        game.food.y < game.head.y,
        game.food.y > game.head.y,
    ]

    return np.array(state, dtype=np.float32)

def select_action(model: LinearQNet, state: np.ndarray) -> List[int]:
    with torch.no_grad():
        state0 = torch.tensor(state, dtype=torch.float32)
        prediction = model(state0)
        move = int(torch.argmax(prediction).item())

    action = [0, 0, 0]
    action[move] = 1
    return action

def evaluate_single_seed(
    model: LinearQNet,
    seed: int,
    render_flag: bool,
    max_steps_without_food: int,
    render_speed: int,
) -> Tuple[float, float]:
    game = SnakeGameAI(
        render=render_flag,
        seed=int(seed),
        speed=render_speed,
        max_steps_without_food=max_steps_without_food,
    )

    cumulative_reward = 0.0
    game_over = False

    try:
        while not game_over:
            state = get_state(game)
            action = select_action(model, state)
            reward, game_over, _ = game.play_step(action)
            cumulative_reward += reward
    finally:
        game.close()

    return cumulative_reward, float(game.score)


def worker_rollout(task):
    (
        model_index,
        model_state_dict,
        seeds,
        render_flag,
        input_size,
        hidden_size,
        output_size,
        max_steps_without_food,
        render_speed,
    ) = task

    torch.set_num_threads(1)

    model = LinearQNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(model_state_dict)
    model.eval()

    fitness_values = []
    score_values = []

    for seed in seeds:
        fitness, score = evaluate_single_seed(
            model=model,
            seed=int(seed),
            render_flag=render_flag,
            max_steps_without_food=max_steps_without_food,
            render_speed=render_speed,
        )
        fitness_values.append(fitness)
        score_values.append(score)

    return model_index, float(np.mean(fitness_values)), float(np.mean(score_values))

def tournament_select(indices: Sequence[int], fitnesses: Sequence[float], tournament_size: int = 3) -> int:
    k = min(tournament_size, len(indices))
    sampled = random.sample(list(indices), k=k)
    return max(sampled, key=lambda idx: fitnesses[idx])

def checkpoint_model(model: LinearQNet, generation: int, checkpoint_dir: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"best_model_gen_{generation:04d}.pth")
    torch.save(model.state_dict(), path)

def initialize_population(config: EvolutionConfig) -> List[LinearQNet]:
    population = []
    for _ in range(config.population_size):
        model = LinearQNet(config.input_size, config.hidden_size, config.output_size)
        add_gaussian_noise_(model, std=config.initial_noise_std)
        model.eval()
        population.append(model)
    return population

def evaluate_population(
    population: Sequence[LinearQNet],
    config: EvolutionConfig,
    seeds: list[int],
) -> Tuple[Sequence[float], np.ndarray]:
    tasks = []
    for idx, model in enumerate(population):
        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        tasks.append(
            (
                idx,
                state_dict,
                list(seeds), # pass a copy of the seeds list to each worker
                config.render_first_worker and idx == 0,
                config.input_size,
                config.hidden_size,
                config.output_size,
                config.max_steps_without_food,
                config.render_speed,
            )
        )

    ctx = mp.get_context("spawn")
    workers = max(1, min(config.num_workers, len(tasks)))

    with ctx.Pool(processes=workers) as pool:
        results = pool.map(worker_rollout, tasks)

    results.sort(key=lambda x: x[0])

    fitnesses = list(map(float, np.array([item[1] for item in results], dtype=np.float32)))
    scores = np.array([item[2] for item in results], dtype=np.float32)
    return fitnesses, scores

def clone_model_from_state(config: EvolutionConfig, state_dict: Dict[str, torch.Tensor]) -> LinearQNet:
    child = LinearQNet(config.input_size, config.hidden_size, config.output_size)
    child.load_state_dict(state_dict)
    child.eval()
    return child

def breed_next_generation(
    population: Sequence[LinearQNet],
    fitnesses: Sequence[float],
    config: EvolutionConfig,
) -> List[LinearQNet]:
    pop_size = len(population)
    elite_count = max(1, int(pop_size * config.elite_fraction))
    mating_count = max(2, int(pop_size * config.mating_fraction))

    ranked_indices = list(np.argsort(fitnesses)[::-1])

    elites = ranked_indices[:elite_count]
    mating_pool = ranked_indices[:mating_count]

    new_population: List[LinearQNet] = []

    for elite_idx in elites:
        elite_state = clone_state_dict(population[elite_idx].state_dict())
        new_population.append(clone_model_from_state(config, elite_state))

    offspring_needed = pop_size - elite_count

    for _ in range(offspring_needed):
        parent_a = tournament_select(mating_pool, fitnesses)
        parent_b = tournament_select(mating_pool, fitnesses)
        while parent_b == parent_a:
            parent_b = tournament_select(mating_pool, fitnesses)

        parent_a_state = population[parent_a].state_dict()
        parent_b_state = population[parent_b].state_dict()

        child_state = uniform_crossover(parent_a_state, parent_b_state)
        child_state = mutate_state_dict(
            child_state,
            mutation_rate=config.mutation_rate,
            mutation_std=config.mutation_std,
        )

        new_population.append(clone_model_from_state(config, child_state))

    return new_population

def run_evolution(config: EvolutionConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    population = initialize_population(config)
    plotter = EvolutionPlotter()

    for generation in range(1, config.generations + 1):
        # fixed seeds for this generation: same rollout conditions for all models
        generation_rng = np.random.default_rng(config.seed + generation)
        seeds = list(map(int, generation_rng.integers(0, 2_000_000_000, size=config.games_per_model, endpoint=False)))

        fitnesses, scores = evaluate_population(population, config, seeds)

        generation_max = float(np.max(fitnesses))
        generation_mean = float(np.mean(fitnesses))
        generation_max_score = float(np.max(scores))
        best_idx = int(np.argmax(fitnesses))
        best_score = float(scores[best_idx])

        plotter.update(generation=generation, generation_max=generation_max, generation_mean=generation_mean, generation_max_score=generation_max_score)

        print(
            f"Generation {generation:04d} | "
            f"max_fitness={generation_max:.2f} | mean_fitness={generation_mean:.2f} | "
            f"best_avg_score={best_score:.2f}"
        )

        if generation % config.checkpoint_every == 0:
            checkpoint_model(population[best_idx], generation, config.checkpoint_dir)

        population = breed_next_generation(population, fitnesses, config)

def parse_args() -> EvolutionConfig:
    parser = argparse.ArgumentParser(description="Parallel Evolutionary Reinforcement Learning for Snake")

    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--games-per-model", type=int, default=3)

    parser.add_argument("--elite-fraction", type=float, default=0.05)
    parser.add_argument("--mating-fraction", type=float, default=0.20)

    parser.add_argument("--mutation-rate", type=float, default=0.10)
    parser.add_argument("--mutation-std", type=float, default=0.02)
    parser.add_argument("--initial-noise-std", type=float, default=0.02)

    parser.add_argument("--max-steps-without-food", type=int, default=120)

    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))

    parser.add_argument("--seed", type=int, default=42)

    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument("--render-first-worker", action="store_true", default=True)
    render_group.add_argument("--no-render", action="store_true")

    parser.add_argument("--render-speed", type=int, default=90)

    args = parser.parse_args()

    return EvolutionConfig(
        population_size=args.population_size,
        generations=args.generations,
        games_per_model=args.games_per_model,
        elite_fraction=args.elite_fraction,
        mating_fraction=args.mating_fraction,
        mutation_rate=args.mutation_rate,
        mutation_std=args.mutation_std,
        initial_noise_std=args.initial_noise_std,
        max_steps_without_food=args.max_steps_without_food,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
        seed=args.seed,
        render_first_worker=False if args.no_render else args.render_first_worker,
        render_speed=args.render_speed,
    )

if __name__ == "__main__":
    cfg = parse_args()
    run_evolution(cfg)
