import unittest
from Solver import Solver

class TestSolver(unittest.TestCase):
    def test_is_complete(self):
        game_info = {
            'game': 'sudoku',
            'GRID_LEN': 9,
            'SQUARE_LEN': 3,
        }
        solver = Solver(game_info)
        assignment = {'00': '5', '01': '3', '02': '4', '03': '6', '04': '7', '05': '8', '06': '9', '07': '1', '08': '2'}
        self.assertTrue(solver.is_complete(assignment))

    def test_select_unassigned_variable(self):
        game_info = {
            'game': 'sudoku',
            'GRID_LEN': 9,
            'SQUARE_LEN': 3,
        }
        solver = Solver(game_info)
        assignment = {'00': None, '01': '3', '02': '4'}
        var = solver.select_unassigned_variable(['00', '01', '02'], assignment)
        self.assertEqual(var, '00')

    def test_is_consistent(self):
        game_info = {
            'game': 'sudoku',
            'GRID_LEN': 9,
            'SQUARE_LEN': 3,
        }
        solver = Solver(game_info)
        assignment = {'00': '5', '01': '3', '02': '4', '10': '6', '11': '7'}
        constraints = [solver.alldiff_in_cols_and_rows]
        self.assertTrue(solver.is_consistent(assignment, constraints))

    def test_easy_inference(self):
        game_info = {
            'game': 'sudoku',
            'GRID_LEN': 9,
            'SQUARE_LEN': 3,
        }
        game_data = {
            'variables_found': {
                '00': '5', '01': '3', '04': '7',
                # Add more pre-filled values
            }
        }
        solver = Solver(game_info)
        assignment = solver.easy_inference(solver.CSP)
        self.assertEqual(assignment['00'], '5')
        self.assertEqual(assignment['01'], '3')

    def test_wrong_initial_assignment(self):
        game_info = {
            'game': 'sudoku',
            'GRID_LEN': 9,
            'SQUARE_LEN': 3,
        }
        game_data = {
            'variables_found': {
                '00': '5', '01': '5',  # Invalid because two 5s are in the same row
            }
        }
        solver = Solver(game_info)
        result = solver.solveGame(game_data)
        self.assertEqual(result, 'WRONG_INITIAL_ASSIGNMENT')

if __name__ == '__main__':
    unittest.main()