import numpy as np
import numpy.typing as npt
import torch
from torch import nn


class CellPredictor:
    """Base class for predicting cell states."""

    def predict(self, cell_states: npt.NDArray) -> int:
        """Takes a grid of cell states and predicts the next state of the inner cells.

        Arguments:
            x: Array of cell states. 1.0 means alive, 0.0 means dead. The array
               is assumed to be of shape (width+2, height+2), where (width, height)
               is the true shape  of the grid. The additional cell at each boundary
               should be padding corresponding to the wrapping rules being used. For
               example, the boundaries can be padded with zeroes to assume that all
               cells outside the grid are dead. Alternatively, values can be wrapped
               from the opposite boundary.

        Returns: The next state of all the cells in the grid. 1.0 means alive, 0.0 means dead.
        """
        raise NotImplementedError("Must be overridden in derived class.")


class ExplicitCellPredictor(CellPredictor):
    """Implementation of `CellStatePredictor` that gives the correct prediction, by definition."""

    def predict(self, cell_states: npt.NDArray) -> npt.NDArray[np.int32]:
        width = cell_states.shape[0] - 2
        height = cell_states.shape[1] - 2
        alive = cell_states[1:width+1, 1:height+1] == 1
        alive_neighbours = np.zeros((width, height))
        for x_offset in [-1, 0, 1]:
            for y_offset in [-1, 0, 1]:
                if x_offset == 0 and y_offset == 0:
                    continue
                alive_neighbours += cell_states[1+x_offset:width+1+x_offset, 1+y_offset:height+1+y_offset]
        predictions = (alive_neighbours == 3) | (alive & (alive_neighbours == 2))
        predictions = predictions.astype(np.int32)
        return predictions


class ModelBasedCellPredictor(CellPredictor):
    def __init__(self, model: nn.Module):
        self.model = model

    def predict(self, cell_states: npt.NDArray) -> npt.NDArray[np.int32]:
        # Convert to torch tensor, add batch and channel dimensions
        X = torch.tensor(cell_states, dtype=torch.float).expand((1, 1, -1, -1))
        with torch.no_grad():
            logits = self.model(X)
        logits = logits.numpy().squeeze(0)
        preds = (logits > 0.0).astype(np.int32)
        return preds

