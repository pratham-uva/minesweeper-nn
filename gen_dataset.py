import pickle
import random

from agent import QLearningAgent
from minesweeper import Minesweeper
from utils import create_random_board, Condition


def generate_training_data(episodes, num_train_data, rows, cols, min_bombs, max_bombs, save_path="training_data.pkl"):
    """
    Generates training data for the Minesweeper game.
    NOTE: This scripe is used with the QLearningAgent implemented in HW4

    Args:
        episodes (int): Number of episodes to train the agent.
        num_train_data (int): Number of training data to generate.
        rows (int): Number of rows in the Minesweeper board.
        cols (int): Number of columns in the Minesweeper board.
        min_bombs (int): Minimum number of bombs on the board.
        max_bombs (int): Maximum number of bombs on the board.
        save_path (str): Path to save the training data.
    Returns:
        list: List of training data of the form {"map": bomb_map, "trajectory": trajectory}.
    """
    training_data = []
    while len(training_data) < num_train_data:
        # Randomize the number of bombs between min_bombs and max_bombs
        num_bombs = random.randint(min_bombs, max_bombs)
        bomb_map = create_random_board(rows, cols, num_bombs)
        game = Minesweeper(rows=rows, cols=cols, bomb_map=bomb_map)
        agent = QLearningAgent(game)
        agent.train(episodes=episodes)
        cond, res = agent.play(return_state=True)
        if cond == Condition.WIN:
            training_data.append({"map": bomb_map, "trajectory": res})
            print(f"Training data size: {len(training_data)} / {num_train_data}")

    # Save the training data to a file
    with open(save_path, "wb") as f:
        pickle.dump(training_data, f)
    print(f"Training data generation complete! Data saved to {save_path}")

    return training_data


if __init__ == "__main__":
    generate_training_data(1000, 10000, 3, 3, 3, 3, "training_data_3x3.pkl")
    generate_training_data(1000, 10, 3, 3, 3, 3, "test_data_3x3.pkl")
