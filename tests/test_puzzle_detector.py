import unittest
import cv2
import numpy as np
from unittest.mock import patch, MagicMock
from PuzzleDetector import PuzzleDetector

class TestPuzzleDetector(unittest.TestCase):

    def setUp(self):

        # Define the size of the image
        image_size = (500, 500)
        # Create a list with a mixture of None and NumPy arrays
        my_list = [
            None, None, None, None, None, None, None, None, 
            np.zeros(image_size + (3,), dtype=np.uint8),
            None, None, None, None, None, None, 
            np.zeros(image_size + (3,), dtype=np.uint8),
            np.zeros(image_size + (3,), dtype=np.uint8),
            None, None, None, np.zeros(image_size + (3,), dtype=np.uint8),
            None, None, None, None, None, np.zeros(image_size + (3,), dtype=np.uint8),
            None, None, None, np.zeros(image_size + (3,), dtype=np.uint8),
            np.zeros(image_size + (3,), dtype=np.uint8),
            None, None, None, np.zeros(image_size + (3,), dtype=np.uint8),
            None, np.zeros(image_size + (3,), dtype=np.uint8),
            np.zeros(image_size + (3,), dtype=np.uint8),
            None, None, None, None, None, np.zeros(image_size + (3,), dtype=np.uint8),
            None, None, None, np.zeros(image_size + (3,), dtype=np.uint8),
            np.zeros(image_size + (3,), dtype=np.uint8),
            None, None, None, np.zeros(image_size + (3,), dtype=np.uint8),
            np.zeros(image_size + (3,), dtype=np.uint8),
            None
        ]

        # stars sample image
        # Define the size and shape of the grid sections
        section_sizes = [ 17, 6, 15, 4, 5, 9, 7, 2]
        # Initialize an empty list to hold the list of lists
        grid_sections = []
        # Generate each section
        for size in section_sizes:
            # Create a NumPy array of string identifiers
            # For simplicity, use a sequence of numbers formatted as strings
            section_array = np.array([f'{i:02}' for i in range(size)], dtype='<U2')
            
            # Convert the NumPy array to a list and append to grid_sections
            grid_sections.append(section_array.tolist())

        
        # skyscrapers sample image
        # Number of arrays you want in the list
        num_arrays = 10
        # Generate the list of zero-filled arrays
        zero_arrays_list = [np.zeros(image_size, dtype=np.uint8) for _ in range(num_arrays)]

        # Sample game_info for different puzzle types
        self.sudoku_info = {'game': 'sudoku', 'GRID_LEN': 9, 'SQUARE_LEN': 3}
        self.stars_info = {'game': 'stars', 'GRID_LEN': 8, 'NUM_STARS': 1}
        self.skyscrapers_info = {'game': 'skyscrapers', 'GRID_LEN': 6, 'SQUARE_LEN': 1}

        # Creating a sample empty image for testing
        self.sudoku_sample_image = [item for item in my_list]
        self.stars_sample_image = [section for section in grid_sections]
        self.skyscrapers_sample_image = [array for i, array in enumerate(zero_arrays_list)]
        
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
