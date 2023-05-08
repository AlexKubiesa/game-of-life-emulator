import sys
import time
import argparse

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


def update_grid(grid: npt.NDArray, cell_predictor: CellPredictor, update_mode: str) -> npt.NDArray:
    padded_grid = np.pad(grid, ((1, 1), (1, 1)))
    preds = cell_predictor.predict(padded_grid)
    if update_mode == "normal":
        new_grid = (preds > 0.5).astype(np.int32)
        return new_grid
    if update_mode == "probabilistic":
        randoms = np.random.random_sample(preds.shape)
        new_grid = (randoms < preds).astype(np.int32)
        return new_grid
    if update_mode == "soft":
        new_grid = preds
        return new_grid
    raise ValueError(f"Unrecognised update mode: {update_mode}.")


def draw_grid(screen: pygame.Surface, grid: npt.NDArray[np.float32]) -> None:
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

    grid_color = pygame.Color(31, 31, 31)
    alive_color = pygame.Color(255, 255, 0)

    pygame.draw.rect(
        screen,
        grid_color,
        (grid_x,
         grid_y,
         grid_width,
         grid_height))
    
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            life = grid[x, y]
            if life > 0.0:
                cell_color = grid_color.lerp(alive_color, life)
                pygame.draw.rect(
                    screen,
                    cell_color,
                    (x * cell_size + border_size + grid_x,
                     y * cell_size + border_size + grid_y,
                     cell_size - border_size,
                     cell_size - border_size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictor",
        default="model",
        choices=["model", "explicit"],
        help="The predictor is the component which controls grid updates. 'model' is a machine" +
        " learning model-based predictor. 'explicit' is a predictor with the Game of Life update" +
        " rule explicitly coded in.")
    parser.add_argument(
        "--update-mode",
        default="normal",
        choices=["normal", "probabilistic", "soft"],
        help="The grid update mode determines how the predictor's output update the grid." +
        " 'normal' means the outputs are thresholded to 0 or 1 and the grid is updated with the" +
        " thresholded values. 'probabilistic' means the outputs are interpreted as probabilities" +
        " p, where the cell becomes 1 with probability p and 0 with probability (1-p). 'soft'" +
        " means the outputs are treated as fractional values, with each cell being somewhere" +
        " between 'alive' and 'dead'.")
    args = parser.parse_args()

    grid = gosper_glider
    cell_predictor = get_cell_predictor(args.predictor)

    pygame.init()
    screen = pygame.display.set_mode((600, 400))

    while True:
        if pygame.QUIT in [e.type for e in pygame.event.get()]:
            sys.exit(0)

        screen.fill((0, 0, 0))
        draw_grid(screen, grid)
        grid = update_grid(grid, cell_predictor, args.update_mode)
        pygame.display.flip()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
