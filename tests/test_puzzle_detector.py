import unittest
import cv2
import numpy as np
from unittest.mock import patch, MagicMock
from PuzzleDetector import PuzzleDetector

class TestPuzzleDetector(unittest.TestCase):

    def setUp(self):
        # Sample game_info for different puzzle types
        self.sudoku_info = {'game': 'sudoku', 'GRID_LEN': 9, 'SQUARE_LEN': 3}
        self.stars_info = {'game': 'stars', 'GRID_LEN': 8, 'NUM_STARS': 1}
        self.skyscrapers_info = {'game': 'skyscrapers', 'GRID_LEN': 6, 'SQUARE_LEN': 1}

        # Creating a sample empty image for testing
        self.sample_image = np.zeros((500, 500, 3), dtype=np.uint8)
        self.detector = PuzzleDetector(self.sudoku_info)

    def test_initialization(self):
        self.assertEqual(self.detector.game_info, self.sudoku_info)
        self.assertIsNone(self.detector.grid_digit_images)

    @patch.object(PuzzleDetector, 'findPolygon')
    @patch.object(cv2, 'imshow')
    def test_detectSudokuBoard(self, mock_imshow, mock_findPolygon):
        mock_findPolygon.return_value = (np.array([[[0, 0]], [[0, 100]], [[100, 100]], [[100, 0]]]), self.sample_image)
        self.detector.detectSudokuBoard(self.sample_image)
        
        self.assertEqual(len(self.detector.grid_digit_images), 81)
        mock_imshow.assert_called()

    @patch.object(PuzzleDetector, 'findPolygon')
    @patch.object(cv2, 'imshow')
    def test_detectStarsBoard(self, mock_imshow, mock_findPolygon):
        self.detector.game_info = self.stars_info
        mock_findPolygon.return_value = (np.array([[[0, 0]], [[0, 100]], [[100, 100]], [[100, 0]]]), self.sample_image)
        self.detector.detectStarsBoard(self.sample_image)

        self.assertEqual(len(self.detector.grid_digit_images), self.stars_info['GRID_LEN'])
        mock_imshow.assert_called()

    @patch.object(PuzzleDetector, 'findPolygon')
    @patch.object(cv2, 'imshow')
    def test_detectSkyscrapersBoard(self, mock_imshow, mock_findPolygon):
        self.detector.game_info = self.skyscrapers_info
        mock_findPolygon.return_value = (np.array([[[0, 0]], [[0, 100]], [[100, 100]], [[100, 0]]]), self.sample_image)
        self.detector.detectSkyscrapersBoard(self.sample_image)

        self.assertTrue(len(self.detector.grid_digit_images) > 0)
        mock_imshow.assert_called()

    def test_findPolygon(self):
        polygon, output = self.detector.findPolygon(self.sample_image)
        self.assertIsNotNone(polygon)
        self.assertIsNotNone(output)

    def test_distance(self):
        p1 = (0, 0)
        p2 = (3, 4)
        self.assertEqual(self.detector.distance(p1, p2), 5)

    def test_get_digit_empty_cell(self):
        empty_cell = np.zeros((50, 50, 3), dtype=np.uint8)
        self.assertIsNone(self.detector.get_digit(empty_cell))

    def test_get_digit_with_digit(self):
        digit_cell = np.ones((50, 50, 3), dtype=np.uint8) * 255
        cv2.putText(digit_cell, '5', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        result = self.detector.get_digit(digit_cell)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (50, 50))

    def test_get_stars_areas(self):
        puzzles = [
            ['00', '01', '02'],
            ['10', '11', '12'],
            ['20', '21', '22']
        ]
        areas = self.detector.get_stars_areas(puzzles)
        self.assertEqual(areas, ['00', '01', '02'])

if __name__ == '__main__':
    unittest.main()
