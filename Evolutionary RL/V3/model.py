import torch
import torch.nn as nn
import copy

class SnakeNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class EvolutionaryTrainer:
    @staticmethod
    def initialize_population(pop_size, input_size=11, hidden_size=256, output_size=3):
        population = []
        for _ in range(pop_size):
            model = SnakeNet(input_size, hidden_size, output_size)
            # Add initial noise to break symmetry
            EvolutionaryTrainer.mutate(model, mutation_rate=1.0, noise_std=0.1)
            population.append(model)
        return population

    @staticmethod
    def crossover(parent1, parent2):
        """Uniform crossover using a binary mask."""
        child = SnakeNet(parent1.linear1.in_features, parent1.linear1.out_features, parent2.linear2.out_features)
        child_sd = child.state_dict()
        p1_sd = parent1.state_dict()
        p2_sd = parent2.state_dict()
        
        for key in child_sd:
            # 50/50 chance to take weight from parent 1 or parent 2
            mask = torch.rand(child_sd[key].shape) > 0.5
            child_sd[key] = torch.where(mask, p1_sd[key], p2_sd[key])
            
        child.load_state_dict(child_sd)
        return child

    @staticmethod
    def mutate(model, mutation_rate=0.1, noise_std=0.1):
        """Gaussian mutation applied to network weights."""
        sd = model.state_dict()
        for key in sd:
            # Create a boolean mask for which weights to mutate
            mask = torch.rand(sd[key].shape) < mutation_rate
            # Generate gaussian noise
            noise = torch.randn(sd[key].shape) * noise_std
            # Apply noise only where mask is True
            sd[key] = sd[key] + (mask * noise)
        model.load_state_dict(sd)