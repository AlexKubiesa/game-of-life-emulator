import sys
import time

import pygame
import numpy as np
import numpy.typing as npt
import torch

from example_grids import gosper_glider
from cell_predictors import ExplicitCellPredictor, ModelBasedCellPredictor, CellPredictor
from models import CellPredictorModel


def get_cell_predictor(name: str) -> CellPredictor:
    """Gets the cell predictor with the given name.
    
    'explicit': Explicit rule-based predictor
    'model'   : ML model-based predictor 
    """
    if name == "explicit":
        return ExplicitCellPredictor()
    if name == "model":
        model = CellPredictorModel()
        model.load_state_dict(torch.load("model_weights.pth"))
        return ModelBasedCellPredictor(model)
    raise ValueError(f"Unrecognised cell predictor name: {name}.")


def update_grid(grid: npt.NDArray, cell_predictor: CellPredictor) -> npt.NDArray:
    padded_grid = np.pad(grid, ((1, 1), (1, 1)))
    new_grid = cell_predictor.predict(padded_grid)
    return new_grid


def draw_grid(screen: pygame.Surface, grid: npt.NDArray) -> None:
    screen_aspect_ratio = screen.get_width() / screen.get_height()
    grid_aspect_ratio = grid.shape[0] / grid.shape[1]
    if grid_aspect_ratio > screen_aspect_ratio:
        grid_width = screen.get_width()
        grid_height = grid_width / grid_aspect_ratio
        cell_size = screen.get_width() / grid.shape[0]
    else:
        grid_height = screen.get_height()
        grid_width = grid_height * grid_aspect_ratio
        cell_size = screen.get_height() / grid.shape[1]
        
    border_size = 2
    grid_x = (screen.get_width() - grid_width) / 2
    grid_y = (screen.get_height() - grid_height) / 2

    pygame.draw.rect(
        screen,
        (31, 31, 31),
        (grid_x,
         grid_y,
         grid_width,
         grid_height))
    
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 1:
                pygame.draw.rect(
                    screen,
                    (255, 255, 0),
                    (x * cell_size + border_size + grid_x,
                     y * cell_size + border_size + grid_y,
                     cell_size - border_size,
                     cell_size - border_size))


def main():
    grid = gosper_glider
    cell_predictor = get_cell_predictor("model")

    pygame.init()
    screen = pygame.display.set_mode((600, 400))

    while True:
        if pygame.QUIT in [e.type for e in pygame.event.get()]:
            sys.exit(0)

        screen.fill((0, 0, 0))
        draw_grid(screen, grid)
        grid = update_grid(grid, cell_predictor)
        pygame.display.flip()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
