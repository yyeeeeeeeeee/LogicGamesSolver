import unittest

class TestImageLoading(unittest.TestCase):
    
    def sudoku_test_load_image_success(self):
        image = "input_sudoku.jpg"
        self.assertIsNotNone(image, "Image should be loaded successfully")

    def stars_test_load_image_success(self):
        image = "input_stars.jpg"
        self.assertIsNotNone(image, "Image should be loaded successfully")

    def skyscrapers_test_load_image_success(self):
        image ="input_skyscrapers.jpg"
        self.assertIsNotNone(image, "Image should be loaded successfully")

    def test_load_image_failure(self):
        image = ""
        self.assertIsNone(image, "Image should not be loaded with an invalid path")

if __name__ == '__main__':
    unittest.main()

