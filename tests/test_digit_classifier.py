import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from DigitClassifier import DigitClassifier

class TestDigitClassifier(unittest.TestCase):
    
    @patch('os.path.isfile')
    @patch('tensorflow.keras.models.load_model')
    def test_initialization_with_weights(self, mock_load_model, mock_isfile):
        mock_isfile.return_value = True
        classifier = DigitClassifier(weights_file='model_weights.h5')
        self.assertTrue(classifier.model_built)
        mock_load_model.assert_called_once_with('model_weights.h5')

    def test_get_model_structure(self):
        classifier = DigitClassifier()
        model = classifier.get_model_structure(28, 28, 1, 10)
        self.assertEqual(len(model.layers), 15)
        self.assertEqual(model.output_shape, (None, 10))

    def test_predict_digit_image(self):

        classifier = DigitClassifier()
        classifier.model = MagicMock()
        classifier.model_built = True

        digit_image = np.zeros((28, 28), dtype=np.uint8)  # Grayscale image
        
        prediction = classifier.predictDigitImage(digit_image)

        self.assertIsNotNone(prediction)


    @patch.object(DigitClassifier, 'predictDigitImage')
    def test_analyze_boards(self, mock_predict_digit_image):
        mock_predict_digit_image.side_effect = lambda x: 1 if np.any(x) else 0
        
        classifier = DigitClassifier()
        
        digit_images = [np.ones((28, 28)), np.zeros((28, 28))]
        info = {'GRID_LEN': 2}
        
        board_structure = classifier.analyze_boards(digit_images, info)
        
        self.assertEqual(board_structure, {'00': '1', '01': '0'})

    def test_save_puzzle(self):
        classifier = DigitClassifier()
        
        puzzle_1 = np.zeros((28, 28))
        puzzle_2 = np.ones((28, 28))
        
        classifier.save_puzzle(puzzle_1)
        classifier.save_puzzle(puzzle_2)
        
        self.assertEqual(len(classifier.puzzles), 2)
        self.assertEqual(classifier.puzzles_seen, 2)

    @patch.object(DigitClassifier, 'analyze_boards')
    def test_get_sudoku_digits(self, mock_analyze_boards):
        mock_analyze_boards.return_value = {'00': '1', '01': '2'}
        
        classifier = DigitClassifier()
        classifier.puzzles = [np.zeros((28, 28)), np.ones((28, 28))]
        classifier.puzzles_seen = 2
        
        info = {'GRID_LEN': 9}
        
        digits_found = classifier.get_sudoku_digits(info)
        
        self.assertEqual(digits_found, {'00': '1', '01': '2'})

    @patch.object(DigitClassifier, 'analyze_skyscrapers_boards')
    def test_get_skyscrapers_digits(self, mock_analyze_skyscrapers_boards):
        mock_analyze_skyscrapers_boards.return_value = {'00': '1', '01': '2'}
        
        classifier = DigitClassifier()
        classifier.puzzles = [np.zeros((28, 28)), np.ones((28, 28))]
        classifier.puzzles_seen = 2
        
        info = {'GRID_LEN': 4}
        
        digits_found = classifier.get_skyscrapers_digits(info)
        
        self.assertEqual(digits_found, {'00': '1', '01': '2'})


if __name__ == '__main__':
    unittest.main()