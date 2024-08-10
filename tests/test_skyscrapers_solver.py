# tests/test_skyscrapers_solver.py
import sys
import pytest
from ..Solver import Solver

def test_solve_valid_skyscrapers():

    info = {
    'game': 'skyscrapers', # sudoku, stars, skyscrapers
    'GRID_LEN': 8,
    'SQUARE_LEN': 2,
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
        [0, 0, 2, 0],
        [0, 0, 0, 1],
        [0, 3, 0, 0],
        [0, 0, 0, 0]
    ]
    expected_solution = [
        [1, 4, 2, 3],
        [2, 3, 4, 1],
        [4, 1, 3, 2],
        [3, 2, 1, 4]
    ]

    solution = solver.solveGame(puzzle)
    assert solution == expected_solution

def test_invalid_skyscrapers():

    info = {
    'game': 'skyscrapers', # sudoku, stars, skyscrapers
    'GRID_LEN': 8,
    'SQUARE_LEN': 2,
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
        [0, 0, 2, 0],
        [0, 0, 0, 1],
        [0, 3, 0, 0],
        [0, 0, 0, 0]
    ]
    with pytest.raises(ValueError):
        solver.solveGame(puzzle)
