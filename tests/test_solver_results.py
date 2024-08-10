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

        if len(sys.argv) > 1:#TODO check + controllo games
            if sys.argv[1] is not None:
                info['game'] = sys.argv[1]
            if sys.argv[2] is not None:
                info['GRID_LEN'] = int(sys.argv[2])
            if len(sys.argv) > 3 and sys.argv[3] is not None:
                info['SQUARE_LEN'] = int(sys.argv[3])

        self.solver = Solver(info)

    @patch('Solver.cv2.putText')
    def test_drawSudokuResult(self, mock_putText):

        self.solver.CSP = {'VARIABLES': [f'cell_{i}' for i in range(self.solver.GRID_LEN * self.solver.GRID_LEN)]}

        # Create a mock grid image
        grid_image = np.zeros((450, 450, 3), dtype=np.uint8)  # 450x450 is arbitrary for testing

        # Mock data for sudoku values
        sudoku_values = {f'cell_{i}': str(i % 9 + 1) for i in range(self.solver.GRID_LEN * self.solver.GRID_LEN)}

        # Run the function
        result_image = self.solver.drawSudokuResult(grid_image, sudoku_values)

        # Check that cv2.putText was called
        self.assertTrue(mock_putText.called)
        self.assertEqual(mock_putText.call_count, self.solver.GRID_LEN * self.solver.GRID_LEN)

    @patch('Solver.cv2.putText')
    def test_drawStarsResult(self, mock_putText):

        # Create a mock grid image
        grid_image = np.zeros((450, 450, 3), dtype=np.uint8)

        # Mock data for stars values
        stars_values = {f'cell_{i}': '0' if i % 2 == 0 else '1' for i in range(self.solver.GRID_LEN * self.solver.GRID_LEN)}

        # Run the function
        result_image = self.solver.drawStarsResult(grid_image, stars_values)

        # Check that cv2.putText was called
        self.assertTrue(mock_putText.called)
        self.assertEqual(mock_putText.call_count, self.solver.GRID_LEN * self.solver.GRID_LEN)

    @patch('Solver.cv2.putText')
    def test_drawSkyscrapersResult(self, mock_putText):

        # Create a mock grid image
        grid_image = np.zeros((450, 450, 3), dtype=np.uint8)

        # Mock data for skyscrapers values
        skyscrapers_values = {f'cell_{i}': str((i % 9) + 1) for i in range(self.solver.GRID_LEN * self.solver.GRID_LEN)}

        # Run the function
        result_image = self.solver.drawSkyscrapersResult(grid_image, skyscrapers_values)

        # Check that cv2.putText was called
        self.assertTrue(mock_putText.called)
        self.assertEqual(mock_putText.call_count, self.solver.GRID_LEN * self.solver.GRID_LEN)

if __name__ == '__main__':
    unittest.main()
