import os
import threading
from optparse import OptionParser

from agent import *
from dataset import MinesweeperDataset, SimplePerceptronDataset
from minesweeper import Minesweeper
from utils import read_bomb_map, read_clue_map

def main():
    """
    Main function to parse command-line options and start the Minesweeper game with the specified agent.

    Command-line options:
    -a, --agent: Type of agent to use (manual or rule_based)
    -m, --map: Path to the bomb map file
    -c, --clue: Path to the clue map file (optional)
    -b, --batch-size: Batch size for training the neural network agent
    -e, --epochs: Number of epochs for training the neural network agent
    -l, --learning-rate: Learning rate for training the neural network agent
    -s, --save-model-path: Path to save the trained model (optional)
    -p, --load-model-path: Path to the pre-trained model file (optional)
    """
    parser = OptionParser()
    parser.add_option("-a", "--agent", dest="agent_type", help="Type of agent to use (manual|nn)")
    parser.add_option("-m", "--map", dest="bomb_map_file", help="Path to the bomb map file")
    parser.add_option("-c", "--clue", dest="clue_map_file", help="Path to the clue map file")
    parser.add_option("-b", "--batch-size", dest="batch_size", type="int", default=8, help="Batch size for training the neural network agent (default: 8)")
    parser.add_option("-e", "--epochs", dest="epochs", type="int", default=200, help="Number of epochs for training the neural network agent (default: 200)")
    parser.add_option("-l", "--learning-rate", dest="learning_rate", type="float", default=0.01, help="Learning rate for training the neural network agent (default: 0.01)")
    parser.add_option("-s", "--save-model-path", dest="save_model_path", help="Path to save the trained model (optional)")
    parser.add_option("-p", "--load-model-path", dest="load_model_path", help="Path to the pre-trained model file (optional)")

    (options, args) = parser.parse_args()

    if not options.agent_type:
        parser.print_help()
        return

    agent_type = options.agent_type.lower()
    bomb_map_file = options.bomb_map_file
    clue_map_file = options.clue_map_file

    bomb_map = None
    if bomb_map_file and  os.path.exists(bomb_map_file):
        bomb_map = read_bomb_map(bomb_map_file)
        rows, cols = len(bomb_map), len(bomb_map[0])

    clue_map = None
    if clue_map_file:
        if not os.path.exists(clue_map_file):
            print(f"Error: The clue map file '{clue_map_file}' does not exist.")
            return
        clue_map = read_clue_map(clue_map_file)

    if agent_type == "manual":
        game = Minesweeper(rows=rows, cols=cols, bomb_map=bomb_map, gui=True)
        agent = ManualGuiAgent(game)
    elif agent_type == "perceptron":
        model = PerceptronModel(dimensions=3)
        dataset = SimplePerceptronDataset(model)
        model.train(dataset)
    elif agent_type == "nn":
        game = Minesweeper(rows=rows, cols=cols, bomb_map=bomb_map)
        agent = NeuralNetworkAgent(game, clue_map, input_size=9, output_size=9, learning_rate=options.learning_rate)
        dataset = MinesweeperDataset("training_data_3x3.pkl")
        if options.load_model_path:
            print("Loading the pre-trained model...")
            agent.load_params(options.load_model_path)
        else:
            print("Training the neural network agent...")
            agent.train(dataset, batch_size=options.batch_size,
                        epochs=options.epochs,
                        save_path=options.save_model_path)
        print("Playing the predefined map with the trained neural network agent.")
        agent.play(reveal_clue=True)
        print("Test with 10 new bomb maps.")
        with open("test_data_3x3.pkl", "rb") as f:
            test_data = pickle.load(f)
        for datum in test_data[:10]:
            bomb_map = datum["map"]
            trajectory = datum["trajectory"]
            game.set_new_board(bomb_map)
            for traj in trajectory[:3]:
                state, action, update_state = traj
                game.step(action)
            agent.play()
    else:
        print("Unknown agent type. Use 'manual' or 'nn'.")
        return

    if agent_type != "perceptron":
        agent_thread = threading.Thread(target=agent.play)
        agent_thread.start()
        if game.gui:
            game.gui.start_gui()

if __name__ == "__main__":
    main()
