from utils import ActionType, Action, Condition

import nn

import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from copy import deepcopy


class Agent:
    def __init__(self, game):
        """
        Initializes the agent with the given game instance.

        Args:
            game: An instance of the game that the agent will interact with.
        """
        self.game = game
        self.rows = game.rows
        self.cols = game.cols

    def play(self):
        """
        Executes the game loop for the agent.

        The agent continuously observes the game state, determines the next action,
        and performs the action until the game reaches a terminal condition.

        Returns:
            goal_test (Condition): The final state of the game, indicating whether
                                   the game is still in progress, won, or reveal a bomb.
        """
        raise NotImplementedError()

    def get_neighbors(self, x, y):
        """
        Get the neighboring coordinates of a given cell in a board.

        Args:
            x (int): The x-coordinate of the cell.
            y (int): The y-coordinate of the cell.

        Returns:
            list of tuple: A list of tuples representing the coordinates of the neighboring cells.
                           Only includes neighbors that are within the bounds of the board.
        """
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        return [(x + dx, y + dy) for dx, dy in directions if 0 <= x + dx < self.game.rows and 0 <= y + dy < self.game.cols]


class ManualGuiAgent(Agent):
    def __init__(self, game):
        super().__init__(game)

    def play(self):
        pass


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def forward(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Args:
            x: a node with shape (1 x dimensions)

        Returns:
            a node containing a single number (the score)
        """
        # TODO: Calculate the score using the weights and input
        raise NotImplementedError()

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        # TODO: Compute the predicted class using the forward method
        raise NotImplementedError()

    def train(self, dataset, visualize=True):
        """
        Train the perceptron until convergence.
        """
        batch_size = 1
        
        # TODO: Implement the training loop
        raise NotImplementedError()

        val_accuracy = dataset.get_validation_accuracy(self)
        print("Final validation accuracy:", val_accuracy)
        if visualize:
            plt.show()


class NeuralNetworkAgent(Agent):
    def __init__(self, game, clues, input_size, output_size, learning_rate=0.001):
        """
        Initializes the Neural Network Agent.

        Args:
            game: An instance of the game that the agent will interact with.
            input_size (int): The size of the input layer (flattened state size).
            output_size (int): The size of the output layer (number of possible actions).
            learning_rate (float): The learning rate for the optimizer.
        """
        super().__init__(game)
        self.clues = clues

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # TODO: Initialize parameters (weights and biases) for each layer of the network
        # Example: self.params = {"w1": ..., "b1": ..., ...}
        raise NotImplementedError()

    def load_params(self, path):
        """
        Load the parameters from a file.

        Args:
            path (str): The path to the file containing the parameters.
        """
        with open(path, "rb") as f:
            self.params = pickle.load(f)

    def reveal_initial_clues(self):
        """
        Reveals the initial clues on the board.

        Iterates through the clues and reveals each clue that is an integer.
        """
        for x in range(len(self.clues)):
            for y in range(len(self.clues[0])):
                clue = self.clues[x][y]
                if isinstance(clue, int):
                    self.game.step(Action(ActionType.REVEAL, x, y))
    
    def forward(self, x):
        """
        Performs a forward pass through the network.

        Your model should predict a node with shape (batch_size x row x col),
        containing scores. Higher scores correspond to greater probability of
        the cell should be revealed.

        Inputs:
            x: a node with shape (batch_size x hidden_size)
        Output:
            A node with shape (batch_size x rol x col) containing predicted scores
                (also called logits)
        """

        # TODO: Implement the forward pass through the network
        # Example: Apply Linear transformation and activation functions, etc.
        raise NotImplementedError()
    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x rol x col).

        Inputs:
            x: a node with shape (batch_size x input_size)
            y: a node with shape (batch_size x rol x col)
        Returns: a loss node
        """
        predicted_scores = self.forward(x)

        # TODO: Compute the loss for classification problem
        raise NotImplementedError()

        return loss
    
    def train(self, dataset, batch_size=8, epochs=10, save_path=""):
        """
        Trains the model using the provided dataset.

        Args:
            dataset: An object that provides training and validation data.
            batch_size (int, optional): The number of samples per batch. Defaults to 8.
            epochs (int, optional): The number of training epochs. Defaults to 10.
            save_path (str, optional): The file path to save the trained parameters.
                                       If empty, the parameters are not saved. Defaults to "".

        Updates:
            - The model's parameters (`self.params`) are updated using gradient descent.
            - The best parameters (based on validation accuracy) are restored after training.

        Notes:
            - The method assumes the model has attributes `self.params` (a dictionary of 
              parameters) and `self.learning_rate` (a float for the learning rate).
            - The `nn.gradients` function is used to compute gradients of the loss with 
              respect to the model's parameters.
        """
        best_val_accuracy = 0.0
        best_params = None
        for epoch in range(epochs):
            iter = 0
            for x, y in dataset.iterate_once(batch_size):
                # Compute the loss
                loss = self.get_loss(x, y)

                # TODO: Compute gradients and update the weights/biases
                raise NotImplementedError()

                val_accuracy = dataset.get_validation_accuracy(self)
                iter += 1
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_params = deepcopy(self.params)
                    print("\tSaved best parameters at epoch", epoch,
                          ", iteration", iter,
                          ", validation accuracy:", val_accuracy)
            print("Epoch:", epoch, ", Validation accuracy:", val_accuracy)

        # Restore the best parameters
        self.params = best_params
        print("Training complete! Best validation accuracy:", best_val_accuracy)
        if save_path != "":
            with open(save_path, "wb") as f:
                pickle.dump(self.params, f)
            print("Trained parameters saved to", save_path)

    def get_action(self, state):
        """
        Chooses an action based on the neural network's predictions.

        Args:
            state (list of list): The current state of the game.

        Returns:
            Action: The chosen action.
        """
        # Flatten the state
        state_values = [
            cell.value if hasattr(cell, 'value') else cell
            for row in state for cell in row
        ]
        # Get the predicted action
        state_values = np.array([state_values], dtype=float)
        predictions = self.forward(nn.Constant(state_values)).data

        # TODO: Get the action with the highest logits score
        raise NotImplementedError()

        # Convert the action index back to an Action object
        x = action_index // self.cols
        y = action_index % self.cols
        return Action(ActionType.REVEAL, x, y)

    def play(self, reveal_clue=False):
        """
        Plays the game using the trained neural network.

        Returns:
            Condition: The final state of the game, indicating whether
                       the game is still in progress, won, or reveal a bomb.
        """
        print("Playing Minesweeper using Neural Network agent.")
        state = self.game.reset()
        if reveal_clue:
            self.reveal_initial_clues()
        condition = Condition.IN_PROGRESS
        while condition == Condition.IN_PROGRESS:
            action = self.get_action(state)
            state, condition = self.game.step(action)
        print("Final game condition:", condition)
        return condition
