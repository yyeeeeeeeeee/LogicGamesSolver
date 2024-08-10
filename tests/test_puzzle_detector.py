import unittest
import cv2
import numpy as np
from unittest.mock import patch, MagicMock
from PuzzleDetector import PuzzleDetector

class TestPuzzleDetector(unittest.TestCase):

    def setUp(self):

        # Define the size of the image
        image_size = (500, 500)

        # Initialize a sample image with zeros (black image)
        self.sudoku_sample_image = np.zeros(image_size + (3,), dtype=np.uint8)
        
        # stars sample image - assuming it represents grid sections as a single image
        section_sizes = [17, 6, 15, 4, 5, 9, 7, 2]
        grid_sections = []
        for size in section_sizes:
            # Just for illustration, we'll create a simple image of each section
            section_image = np.full((size, size, 3), 255, dtype=np.uint8)  # white square of the given size
            grid_sections.append(section_image)
        # Combine sections into a single image if needed (this is an example, adjust as necessary)
        self.stars_sample_image = np.vstack(grid_sections)
        
        # Skyscrapers sample image
        self.skyscrapers_sample_image = np.zeros(image_size, dtype=np.uint8) # Use a single zero-filled array

        # Sample game_info for different puzzle types
        self.sudoku_info = {'game': 'sudoku', 'GRID_LEN': 9, 'SQUARE_LEN': 3}
        self.stars_info = {'game': 'stars', 'GRID_LEN': 8, 'NUM_STARS': 1}
        self.skyscrapers_info = {'game': 'skyscrapers', 'GRID_LEN': 6, 'SQUARE_LEN': 1}
        
        # Initialize PuzzleDetector instances
        self.sudoku_detector = PuzzleDetector(self.sudoku_info)
        self.stars_detector = PuzzleDetector(self.stars_info)
        self.skyscrapers_detector = PuzzleDetector(self.skyscrapers_info)

    def test_initialization(self):
        self.assertEqual(self.sudoku_detector.game_info, self.sudoku_info)
        self.assertIsNone(self.sudoku_detector.grid_digit_images)

    @patch.object(PuzzleDetector, 'findPolygon')
    @patch.object(cv2, 'imshow')
    def test_detectSudokuBoard(self, mock_imshow, mock_findPolygon):
        mock_findPolygon.return_value = (np.array([[[0, 0]], [[0, 100]], [[100, 100]], [[100, 0]]]), self.sudoku_sample_image)
        self.sudoku_detector.detectSudokuBoard(self.sudoku_sample_image)
        
        self.assertEqual(len(self.detector.grid_digit_images), 81)
        mock_imshow.assert_called()

    @patch.object(PuzzleDetector, 'findPolygon')
    @patch.object(cv2, 'imshow')
    def test_detectStarsBoard(self, mock_imshow, mock_findPolygon):
        self.stars_detector.game_info = self.stars_info
        mock_findPolygon.return_value = (np.array([[[0, 0]], [[0, 100]], [[100, 100]], [[100, 0]]]), self.stars_sample_image)
        self.stars_detector.detectStarsBoard(self.stars_sample_image)

        self.assertEqual(len(self.stars_detector.grid_digit_images), self.stars_info['GRID_LEN'])
        mock_imshow.assert_called()

    @patch.object(PuzzleDetector, 'findPolygon')
    @patch.object(cv2, 'imshow')
    def test_detectSkyscrapersBoard(self, mock_imshow, mock_findPolygon):
        self.detector.game_info = self.skyscrapers_info
        mock_findPolygon.return_value = (np.array([[[0, 0]], [[0, 100]], [[100, 100]], [[100, 0]]]), self.skyscrapers_sample_image)
        self.skyscrapers_detector.detectSkyscrapersBoard(self.skyscrapers_sample_image)

        self.assertTrue(len(self.skyscrapers_detector.grid_digit_images) > 0)
        mock_imshow.assert_called()

    def test_findPolygon(self):
        polygon, output = self.sudoku_detector.findPolygon(self.sudoku_sample_image)
        self.assertIsNotNone(polygon)
        self.assertIsNotNone(output)

    def test_distance(self):
        p1 = (0, 0)
        p2 = (3, 4)
        self.assertEqual(self.sudoku_detector.distance(p1, p2), 5)

    def test_get_digit_empty_cell(self):
        empty_cell = np.zeros((50, 50, 3), dtype=np.uint8)
        self.assertIsNone(self.sudoku_detector.get_digit(empty_cell))

    def test_get_digit_with_digit(self):
        digit_cell = np.ones((50, 50, 3), dtype=np.uint8) * 255
        cv2.putText(digit_cell, '5', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        result = self.sudoku_detector.get_digit(digit_cell)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (50, 50))

    def test_get_stars_areas(self):
        puzzles = [
            ['00', '01', '02'],
            ['10', '11', '12'],
            ['20', '21', '22']
        ]
        areas = self.stars_detector.get_stars_areas(puzzles)
        self.assertEqual(areas, ['00', '01', '02'])

if __name__ == '__main__':
    unittest.main()
