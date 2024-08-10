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

    @patch('tensorflow.keras.preprocessing.image.img_to_array')
    @patch('tensorflow.keras.models.Sequential.predict')
    def test_predict_digit_image(self, mock_predict, mock_img_to_array):
        mock_img_to_array.return_value = np.zeros((28, 28, 1))
        mock_predict.return_value = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])

        classifier = DigitClassifier()
        classifier.model = MagicMock()
        classifier.model_built = True

        #digit_image = np.zeros((28, 28), dtype=np.uint8)  # Grayscale image
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
        
        prediction = classifier.predictDigitImage(sudoku_values)
        print("prediction length: ", len(prediction))
        
        self.assertNotEqual(prediction, 81)

        # Verify that the mocked methods were called as expected
        mock_img_to_array.assert_called_once()
        mock_predict.assert_called_once()

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