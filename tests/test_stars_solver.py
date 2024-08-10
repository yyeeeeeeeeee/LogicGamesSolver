# tests/test_stars_solver.py
import pytest
import sys
from ..Solver import Solver

def test_solve_valid_stars():

    info = {
    'game': 'stars', # sudoku, stars, skyscrapers
    'GRID_LEN': 8,
    'SQUARE_LEN': 1,
    'NUM_STARS': 1
   }

    if len(sys.argv) > 1:#TODO check + controllo games
        if sys.argv[1] is not None:
            info['game'] = sys.argv[1]
        if sys.argv[2] is not None:
            info['GRID_LEN'] = int(sys.argv[2])
        if len(sys.argv) > 3 and sys.argv[3] is not None:
            info['SQUARE_LEN'] = int(sys.argv[3])

    solver = Solver(info)

    puzzle = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    # Example of expected solution with stars represented as 1s
    expected_solution = [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1]
    ]
    solution = solver.solveGame(puzzle)
    assert solution == expected_solution

def test_invalid_stars():
    info = {
    'game': 'stars', # sudoku, stars, skyscrapers
    'GRID_LEN': 8,
    'SQUARE_LEN': 1,
    'NUM_STARS': 1
   }

    if len(sys.argv) > 1:#TODO check + controllo games
        if sys.argv[1] is not None:
            info['game'] = sys.argv[1]
        if sys.argv[2] is not None:
            info['GRID_LEN'] = int(sys.argv[2])
        if len(sys.argv) > 3 and sys.argv[3] is not None:
            info['SQUARE_LEN'] = int(sys.argv[3])

    solver = Solver(info)

    puzzle = [
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 0]
    ]
    with pytest.raises(ValueError):
        solver.solveGame(puzzle)

def test_empty_stars_grid():

    info = {
    'game': 'stars', # sudoku, stars, skyscrapers
    'GRID_LEN': 8,
    'SQUARE_LEN': 1,
    'NUM_STARS': 1
   }

    if len(sys.argv) > 1:#TODO check + controllo games
        if sys.argv[1] is not None:
            info['game'] = sys.argv[1]
        if sys.argv[2] is not None:
            info['GRID_LEN'] = int(sys.argv[2])
        if len(sys.argv) > 3 and sys.argv[3] is not None:
            info['SQUARE_LEN'] = int(sys.argv[3])

    solver = Solver(info)

    puzzle = [
        [0, 0],
        [0, 0]
    ]
    expected_solution = [
        [1, 0],
        [0, 1]
    ]
    solution = solver.solveGame(puzzle)
    assert solution == expected_solution
