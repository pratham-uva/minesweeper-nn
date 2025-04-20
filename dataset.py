from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

import nn


class Dataset(object):
    def __init__(self, x, y):
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert np.issubdtype(x.dtype, np.floating)
        assert np.issubdtype(y.dtype, np.floating)
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def iterate_once(self, batch_size):
        assert isinstance(batch_size, int) and batch_size > 0, (
            "Batch size should be a positive integer, got {!r}".format(
                batch_size))
        #assert self.x.shape[0] % batch_size == 0, (
        #    "Dataset size {:d} is not divisible by batch size {:d}".format(
        #        self.x.shape[0], batch_size))
        index = 0
        div = self.x.shape[0] // batch_size
        while index < div * batch_size: #self.x.shape[0]:
            x = self.x[index:index + batch_size]
            y = self.y[index:index + batch_size]
            yield nn.Constant(x), nn.Constant(y)
            index += batch_size

    def iterate_forever(self, batch_size):
        while True:
            yield from self.iterate_once(batch_size)

    def get_validation_accuracy(self):
        raise NotImplementedError(
            "No validation data is available for this dataset. ")
    

class SimplePerceptronDataset(Dataset):
    def __init__(self, model, visualize=True):
        points = 500
        val_points = 50
        x = np.hstack([np.random.randn(points+val_points, 2), np.ones((points+val_points, 1))])
        y = np.where(x[:, 0] + 3 * x[:, 1] - 2 >= 0, 1.0, -1.0)

        self.val_x = x[points:]
        self.val_y = y[points:]
        x = x[:points]
        y = y[:points]
        super().__init__(x, np.expand_dims(y, axis=1))

        self.model = model
        self.epoch = 0
        self.visualize = visualize

        if visualize:
            fig, ax = plt.subplots(1, 1)
            limits = np.array([-3.0, 3.0])
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            positive = ax.scatter(*x[y == 1, :-1].T, color="red", marker="+")
            negative = ax.scatter(*x[y == -1, :-1].T, color="blue", marker="_")
            line, = ax.plot([], [], color="black")
            text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top")
            ax.legend([positive, negative], [1, -1])
            plt.show(block=False)

            self.fig = fig
            self.limits = limits
            self.line = line
            self.text = text
            self.last_update = time.time()

    def iterate_once(self, batch_size):
        self.epoch += 1

        for i, (x, y) in enumerate(super().iterate_once(batch_size)):
            yield x, y

            if self.visualize and time.time() - self.last_update > 0.01:
                w = self.model.get_weights().data.flatten()
                limits = self.limits
                if w[1] != 0:
                    self.line.set_data(limits, (-w[0] * limits - w[2]) / w[1])
                elif w[0] != 0:
                    self.line.set_data(np.full(2, -w[2] / w[0]), limits)
                else:
                    self.line.set_data([], [])
                self.text.set_text(
                    "epoch: {:,}\npoint: {:,}/{:,}\nweights: {}".format(
                        self.epoch, i * batch_size + 1, len(self.x), w))
                self.fig.canvas.draw_idle()
                self.fig.canvas.start_event_loop(1e-3)
                self.last_update = time.time()

    def iterate_validation(self):
        for i in range(len(self.val_x)):
            yield nn.Constant(self.val_x[i:i + 1]), nn.Constant(self.val_y[i:i + 1])

    def get_validation_accuracy(self, model):
        correct = 0
        for x, y in self.iterate_validation():
            if model.get_prediction(x) == nn.as_scalar(y):
                correct += 1
        return correct / len(self.val_x.data)


class MinesweeperDataset(Dataset):
    def __init__(self, pickle_file_path, validation_split=0.01):
        """
        Initializes the Minesweeper dataset by loading trajectories from a pickle file.

        Args:
            pickle_file_path (str): Path to the pickle file containing the trajectories.
        """
        # Load the data from the pickle file
        with open(pickle_file_path, "rb") as f:
            data = pickle.load(f)

        # Extract states and actions from the trajectories
        states = []
        actions = []
        actions_raw = []
        for datum in data:
            trajectory = datum["trajectory"]
            for t in range(len(trajectory)):
                if t < 4:  # skip the first three steps as they are not discremative
                    continue
                state, action, _ = trajectory[t]  # Each trajectory is (state, action, updated_state)
                # Convert the state (list of lists of Cell) to a flattened list of values
                state_values = [
                    cell.value if hasattr(cell, 'value') else cell
                    for row in state for cell in row
                ]
                # Convert the action to a single value that indicates the cell to reveal
                action_index = action.x * len(state) + action.y
                action_one_hot = np.zeros(len(state)*len(state), dtype=float)
                action_one_hot[action_index] = 1.0
                states.append(state_values)
                actions.append(action_one_hot)
                actions_raw.append(action_index)

        # Convert states and actions to numpy arrays
        states = np.array(states, dtype=float)  # States
        actions = np.array(actions, dtype=float)  # Actions
        actions_raw = np.array(actions_raw)  # Raw actions
     
        # Shuffle and split the data into training and validation sets
        data_size = len(states)
        indices = np.arange(data_size)
        np.random.shuffle(indices)

        split_index = int(data_size * (1 - validation_split))
        train_indices = indices[:split_index]
        val_indices = indices[split_index:]

        # Create training and validation sets
        # Convert states and actions to numpy arrays
        x = states[train_indices]
        y = actions[train_indices]
        self.y_raw = actions_raw[train_indices]
        self.val_x = states[val_indices]
        self.val_y = actions_raw[val_indices]

        # Initialize the parent Dataset class
        super().__init__(x, y)

    def iterate_once(self, batch_size):
        """
        Iterates over the dataset once in batches.

        Args:
            batch_size (int): The size of each batch.

        Yields:
            tuple: A tuple (x, y) where x is the batch of states and y is the batch of actions.
        """
        for x, y in super().iterate_once(batch_size):
            yield x, y

    def get_validation_accuracy(self, model):
        dev_logits = model.forward(nn.Constant(self.val_x)).data
        dev_predicted = np.argmax(dev_logits, axis=1)
        dev_accuracy = np.mean(dev_predicted == self.val_y)
        return dev_accuracy
