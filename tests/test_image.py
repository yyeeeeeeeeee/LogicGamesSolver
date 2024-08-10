from unittest.mock import patch
import unittest
from PIL import Image
from your_module import load_image  # Replace with actual module import

class TestImageLoading(unittest.TestCase):
    
    def sudoku_test_load_image_success(self):
        image = load_image("input_sudoku.jpg")
        self.assertIsNotNone(image, "Image should be loaded successfully")
        self.assertIsInstance(image, Image.Image, "Loaded object should be an instance of PIL.Image.Image")

    def stars_test_load_image_success(self):
        image = load_image("input_stars.jpg")
        self.assertIsNotNone(image, "Image should be loaded successfully")
        self.assertIsInstance(image, Image.Image, "Loaded object should be an instance of PIL.Image.Image")

    def skyscrapers_test_load_image_success(self):
        image = load_image("input_skyscrapers.jpg")
        self.assertIsNotNone(image, "Image should be loaded successfully")
        self.assertIsInstance(image, Image.Image, "Loaded object should be an instance of PIL.Image.Image")
    
    def test_load_image_failure(self):
        image = load_image("input_.jpg")
        self.assertIsNone(image, "Image should not be loaded with an invalid path")

class TestImageDisplay(unittest.TestCase):
    
    @patch('your_module.tk.Label')  # Mocking Tkinter Label
    @patch('your_module.tk.Tk')     # Mocking Tkinter Tk
    def test_display_image(self, mock_tk, mock_label):
        mock_tk.return_value = mock_tk
        mock_label.return_value = mock_label

        image = Image.new("RGB", (100, 100), color = "red")  # Creating a dummy image
        display_image(image)
        
        mock_tk.assert_called_once()
        mock_label.assert_called_once()

class TestImageProcessing(unittest.TestCase):

    def test_process_image(self):
        image = Image.new("RGB", (100, 100), color = "blue")  # Creating a dummy image
        processed_image = process_image(image)
        
        self.assertIsNotNone(processed_image, "Processed image should not be None")
        self.assertIsInstance(processed_image, Image.Image, "Processed object should be an instance of PIL.Image.Image")
        self.assertEqual(processed_image.mode, "RGB", "Processed image mode should be RGB")

if __name__ == '__main__':
    unittest.main()

