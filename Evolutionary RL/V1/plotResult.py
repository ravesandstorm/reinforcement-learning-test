import matplotlib.pyplot as plt

class EvolutionPlotter:
    def __init__(self):
        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(9, 5))
        self.generation_max_fitnesses = []
        self.generation_mean_fitnesses = []
        self.generation_max_scores = []

    def update(self, generation: int, generation_max: float, generation_mean: float, generation_max_score: float):
        self.generation_max_fitnesses.append(generation_max)
        self.generation_mean_fitnesses.append(generation_mean)
        self.generation_max_scores.append(generation_max_score)

        self.ax.clear()
        self.ax.set_title("Parallel Evolutionary RL Progress")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness (Avg Episodic Reward)")

        x = list(range(1, len(self.generation_max_fitnesses) + 1))
        self.ax.plot(x, self.generation_max_fitnesses, label="Generation Max Fitness")
        self.ax.plot(x, self.generation_mean_fitnesses, label="Generation Mean Fitness")
        self.ax.plot(x, self.generation_max_scores, label="Generation Max Score")
        self.ax.legend(loc="best")

        self.ax.text(x[-1], self.generation_max_fitnesses[-1], f"{self.generation_max_fitnesses[-1]:.2f}")
        self.ax.text(x[-1], self.generation_mean_fitnesses[-1], f"{self.generation_mean_fitnesses[-1]:.2f}")
        self.ax.text(x[-1], self.generation_max_scores[-1], f"{self.generation_max_scores[-1]:.2f}")

        self.figure.tight_layout()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        plt.pause(0.001)
