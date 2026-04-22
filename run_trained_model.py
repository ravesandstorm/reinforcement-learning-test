import argparse
import os
import sys
import torch

from agent import Agent
from main import SnakeGameAI

configs = [
    {
        "INPUT_SIZE": 11,
        "HIDDEN_SIZE": 256,
        "OUTPUT_SIZE": 3
    },
    {
        "INPUT_SIZE": 12,
        "HIDDEN_SIZE": 256,
        "OUTPUT_SIZE": 3
    },
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained Snake DQN model (.pth) in inference mode")
    parser.add_argument(
        "--model-path",
        default=os.path.join("model", "model.pth"),
        help="Path to the trained .pth model file (default: model/model.pth)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=0,
        help="Number of games to run before exiting. Use 0 to run indefinitely (default: 0)",
    )
    return parser.parse_args()


def choose_action(agent: Agent, state):
    state0 = torch.tensor(state, dtype=torch.float)
    with torch.no_grad():
        prediction = agent.model(state0)
    move = int(torch.argmax(prediction).item())
    if move < 0:
        move = 0
    elif move > 2:
        move = 2
    final_move = [0, 0, 0]
    final_move[move] = 1
    return final_move


def run_inference(model_path: str, max_games: int = 0) -> None:
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Train first so a checkpoint is saved, or pass --model-path to an existing .pth file.")
        sys.exit(1)

    agent = Agent()

    try:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        agent.model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure the model architecture matches the checkpoint and that the file is a valid .pth model.")
        sys.exit(1)
    agent.model.eval()

    game = SnakeGameAI()
    played_games = 0

    print(f"Loaded model from: {model_path}")
    print("Inference started. Close the game window to stop.")

    while True:
        state_old = agent.get_state(game)
        action = choose_action(agent, state_old)
        _, game_over, score = game.play_step(action)

        if game_over:
            played_games += 1
            print(f"Game {played_games} | Score: {score}")
            game.reset()

            if max_games > 0 and played_games >= max_games:
                print(f"Reached --games={max_games}. Exiting.")
                break


def main() -> None:
    args = parse_args()
    run_inference(args.model_path, args.games)


if __name__ == "__main__":
    main()
