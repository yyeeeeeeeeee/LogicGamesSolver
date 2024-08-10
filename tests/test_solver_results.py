import unittest
from unittest.mock import patch, MagicMock
import sys
import numpy as np
import cv2
from Solver import Solver

class TestSolverResults(unittest.TestCase):

    def setUp(self):
        info = {
            'game': 'sudoku', # sudoku
            'GRID_LEN': 9,
            'SQUARE_LEN': 3,
        }
        if len(sys.argv) > 1:
            try:
                if sys.argv[1] is not None:
                    info['game'] = sys.argv[1]
                if len(sys.argv) > 2 and sys.argv[2].isdigit():
                    info['GRID_LEN'] = int(sys.argv[2])
                if len(sys.argv) > 3 and sys.argv[3].isdigit():
                    info['SQUARE_LEN'] = int(sys.argv[3])
            except (ValueError, IndexError):
                pass  # Ignore errors and use default values

        self.solver = Solver(info)

        cells = []
        [[cells.append(str(i) + str(j)) for j in range(self.solver.GRID_LEN)] for i in range(self.solver.GRID_LEN)]

        domains = {}
        for var in cells:
            domains[var] = [str(k + 1) for k in range(self.solver.GRID_LEN)]

        self.solver.CSP = {
            "VARIABLES": cells,
            "DOMAINS": domains,
            "CONSTRAINTS": [self.solver.alldiff_in_cols_and_rows, self.solver.all_diff_in_areas]
        }


    @patch.object(cv2, 'putText')
    def test_drawSudokuResult(self, mock_putText):

        # Create a mock grid image
        grid_image = np.zeros((450, 450, 3), dtype=np.uint8)  # 450x450 is arbitrary for testing

        # Mock data for sudoku values
        sudoku_values = {'00': '6', '01': '8', '02': '4', '03': '1', '04': '5', '05': '9', '06': '7',
                        '07': '3', '08': '2', '10': '7', '11': '5', '12': '1', '13': '8', '14': '3',
                        '15': '2', '16': '9', '17': '4', '18': '6', '20': '9', '21': '2', '22': '3',
                        '23': '6', '24': '7', '25': '4', '26': '1', '27': '8', '28': '5', '30': '1',
                        '31': '9', '32': '2', '33': '3', '34': '6', '35': '5', '36': '8', '37': '7',
                        '38': '4', '40': '8', '41': '4', '42': '5', '43': '2', '44': '1', '45': '7', 
                        '46': '6', '47': '9', '48': '3', '50': '3', '51': '6', '52': '7', '53': '4', 
                        '54': '9', '55': '8', '56': '2', '57': '5', '58': '1', '60': '2', '61': '3', 
                        '62': '9', '63': '7', '64': '4', '65': '6', '66': '5', '67': '1', '68': '8', 
                        '70': '5', '71': '1', '72': '6', '73': '9', '74': '8', '75': '3', '76': '4', 
                        '77': '2', '78': '7', '80': '4', '81': '7', '82': '8', '83': '5', '84': '2', 
                        '85': '1', '86': '3', '87': '6', '88': '9'}

        # Run the function
        result_image = self.solver.drawSudokuResult(grid_image, sudoku_values)
        print("result_image shape sudoku: ", result_image.shape)

        # Verify the image
        self.assertEqual(result_image.shape, (450, 450, 3))  # Check image size
        self.assertEqual(result_image.dtype, np.uint8)  # Check image type

        # Check that cv2.putText was called
        self.assertTrue(mock_putText.called)
        self.assertEqual(mock_putText.call_count, self.solver.GRID_LEN * self.solver.GRID_LEN)

    @patch.object(cv2, 'putText')
    def test_drawStarsResult(self, mock_putText):

        # Create a mock grid image
        grid_image = np.zeros((450, 450, 3), dtype=np.uint8)

        # Mock data for stars values
        stars_values = {'00': '0', '01': '0', '02': '1', '03': '0', '04': '0', '05': '0', '06': '0', 
                    '07': '0', '10': '0', '11': '0', '12': '0', '13': '0', '14': '0', '15': '0', 
                    '16': '1', '17': '0', '20': '0', '21': '0', '22': '0', '23': '1', '24': '0', 
                    '25': '0', '26': '0', '27': '0', '30': '0', '31': '0', '32': '0', '33': '0', 
                    '34': '0', '35': '0', '36': '0', '37': '1', '40': '1', '41': '0', '42': '0', 
                    '43': '0', '44': '0', '45': '0', '46': '0', '47': '0', '50': '0', '51': '0', 
                    '52': '0', '53': '0', '54': '0', '55': '1', '56': '0', '57': '0', '60': '0', 
                    '61': '1', '62': '0', '63': '0', '64': '0', '65': '0', '66': '0', '67': '0', 
                    '70': '0', '71': '0', '72': '0', '73': '0', '74': '1', '75': '0', '76': '0', 
                    '77': '0'}

        # Run the function
        result_image = self.solver.drawStarsResult(grid_image, stars_values)
        print("result_image shape stars: ", result_image.shape)

        # Verify the image
        self.assertEqual(result_image.shape, (450, 450, 3))  # Check image size
        self.assertEqual(result_image.dtype, np.uint8)  # Check image type

        # Check that cv2.putText was called
        self.assertEqual(mock_putText.call_count, self.solver.GRID_LEN * self.solver.GRID_LEN)
        self.assertTrue(mock_putText.called)
        

    @patch.object(cv2, 'putText')
    def test_drawSkyscrapersResult(self, mock_putText):

        # Create a mock grid image
        grid_image = np.zeros((450, 450, 3), dtype=np.uint8)

        # Mock data for skyscrapers values
        skyscrapers_values = {'01': '2', '02': '1', '03': '6', '04': '3', '05': '2', '06': '2', 
                            '10': '1', '17': '2', '20': '4', '27': '1', '30': '3', '37': '2', 
                            '40': '2', '47': '2', '50': '1', '57': '2'}


        # Run the function
        result_image = self.solver.drawSkyscrapersResult(grid_image, skyscrapers_values)
        print("result_image shape skyscrapers: ", result_image.shape)

        # Verify the image
        self.assertEqual(result_image.shape, (450, 450, 3))  # Check image size
        self.assertEqual(result_image.dtype, np.uint8)  # Check image type

        # Check that cv2.putText was called
        self.assertTrue(mock_putText.called)
        self.assertEqual(mock_putText.call_count, self.solver.GRID_LEN * self.solver.GRID_LEN)

if __name__ == '__main__':
    unittest.main()
