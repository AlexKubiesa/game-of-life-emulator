from collections import namedtuple
import numpy as np
import numpy.typing as npt

Dim = namedtuple("Dimension", ["width", "height"])
Grid = namedtuple("Grid", ["dim", "cells"])
Neighbours = namedtuple("Neighbours", ["alive", "dead"])

def sparse_to_dense(sparse_grid: Grid) -> npt.NDArray:
    dense_grid = np.zeros(sparse_grid.dim)
    for cell in sparse_grid.cells:
        dense_grid[cell[0], cell[1]] = 1
    return dense_grid

def dense_to_sparse(dense_grid: npt.NDArray) -> Grid:
    width, height = dense_grid.shape
    dim = Dim(width, height)
    cells = set()

    for x in range(width):
        for y in range(height):
            if dense_grid[x, y] == 1:
                cells.add((x, y))
    
    return Grid(dim, cells)
