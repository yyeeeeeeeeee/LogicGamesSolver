import unittest
from unittest.mock import patch, MagicMock
from Solver import Solver
from DigitClassifier import DigitClassifier
from PuzzleDetector import PuzzleDetector

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
        solver.iterations = 0

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
        assignment = {'00': '6', '01': '8', '02': '4', '03': '1', '04': '5', '05': '9', 
                      '06': '7', '07': '3', '08': '2', '10': '7', '11': '5', '12': '1', 
                      '13': '8', '14': '3', '15': '2', '16': '9', '17': '4', '18': '6', 
                      '20': '9', '21': '2', '22': '3', '23': '6', '24': '7', '25': '4', 
                      '26': '1', '27': '8', '28': '5', '30': '1', '31': '9', '32': '2', 
                      '33': '3', '34': '6', '35': '5', '36': '8', '37': '7', '38': '4', 
                      '40': '8', '41': '4', '42': '5', '43': '2', '44': '1', '45': '7', 
                      '46': '6', '47': '9', '48': '3', '50': '3', '51': '6', '52': '7', 
                      '53': '4', '54': '9', '55': '8', '56': '2', '57': '5', '58': '1', 
                      '60': '2', '61': '3', '62': '9', '63': '7', '64': '4', '65': '6', 
                      '66': '5', '67': '1', '68': '8', '70': '5', '71': '1', '72': '6', 
                      '73': '9', '74': '8', '75': '3', '76': '4', '77': '2', '78': '7', 
                      '80': '4', '81': '7', '82': '8', '83': '5', '84': '2', '85': '1', 
                      '86': '3', '87': '6', '88': '9'}

        constraints = [solver.alldiff_in_cols_and_rows]
        self.assertTrue(solver.is_consistent(assignment, constraints))


    @patch.object(DigitClassifier, 'get_sudoku_digits')
    @patch.object(DigitClassifier, 'get_skyscrapers_digits')
    @patch.object(PuzzleDetector, 'get_stars_areas')
    def test_easy_inference(self, mock_get_skyscrapers_digits, mock_get_stars_areas, mock_get_sudoku_digits):
        # Set up mocks
        mock_get_sudoku_digits.return_value = {'00': '6', '01': '8', '02': '4', '03': '1', '04': '5', '05': '9', 
                      '06': '7', '07': '3', '08': '2', '10': '7', '11': '5', '12': '1', 
                      '13': '8', '14': '3', '15': '2', '16': '9', '17': '4', '18': '6', 
                      '20': '9', '21': '2', '22': '3', '23': '6', '24': '7', '25': '4', 
                      '26': '1', '27': '8', '28': '5', '30': '1', '31': '9', '32': '2', 
                      '33': '3', '34': '6', '35': '5', '36': '8', '37': '7', '38': '4', 
                      '40': '8', '41': '4', '42': '5', '43': '2', '44': '1', '45': '7', 
                      '46': '6', '47': '9', '48': '3', '50': '3', '51': '6', '52': '7', 
                      '53': '4', '54': '9', '55': '8', '56': '2', '57': '5', '58': '1', 
                      '60': '2', '61': '3', '62': '9', '63': '7', '64': '4', '65': '6', 
                      '66': '5', '67': '1', '68': '8', '70': '5', '71': '1', '72': '6', 
                      '73': '9', '74': '8', '75': '3', '76': '4', '77': '2', '78': '7', 
                      '80': '4', '81': '7', '82': '8', '83': '5', '84': '2', '85': '1', 
                      '86': '3', '87': '6', '88': '9'} 
        mock_get_stars_areas.return_value = {}
        mock_get_skyscrapers_digits.return_value = {}
        
        game_info = {
            'game': 'sudoku', #test for sudoku only
            'GRID_LEN': 9,
            'SQUARE_LEN': 3,
        }
        solver = Solver(game_info)
        classifier = DigitClassifier()

        digits_found = {}
        if game_info['game'] == 'sudoku':
            digits_found = mock_get_sudoku_digits(game_info)
        elif game_info['game'] == 'stars':
            digits_found = mock_get_stars_areas(classifier.puzzles)
        elif game_info['game'] == 'skyscrapers':
            digits_found = mock_get_skyscrapers_digits(game_info)

        # 3. Game solution phase
        data = {
            'variables_found': digits_found
        }
        solver.data = data

        cells = []
        [[cells.append(str(i) + str(j)) for j in range(solver.GRID_LEN)] for i in range(solver.GRID_LEN)]

        domains = {}
        for var in cells:
            domains[var] = [str(k + 1) for k in range(solver.GRID_LEN)]

        solver.CSP = {
            "VARIABLES": cells,
            "DOMAINS": domains,
            "CONSTRAINTS": [solver.alldiff_in_cols_and_rows, solver.all_diff_in_areas]
        }

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
        self.assertEqual(result, 'FAILURE')

if __name__ == '__main__':
    unittest.main()