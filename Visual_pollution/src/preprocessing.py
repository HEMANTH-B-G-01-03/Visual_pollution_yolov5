import cv2
import os

class ImagePreprocessor:
    """
    Image Pre-processing Module
    - Noise Removal using Mean Filtering
    - Contrast Enhancement using CLAHE
    """

    def __init__(self):
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

    def remove_noise(self, image):
        """
        Applies Mean Filtering for noise removal
        """
        return cv2.blur(image, (5, 5))

    def enhance_contrast(self, image):
        """
        Applies CLAHE for contrast enhancement
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l_enhanced = self.clahe.apply(l)

        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def preprocess(self, image_path):
        """
        Complete pre-processing pipeline
        """
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Image not found")

        image = self.remove_noise(image)
        image = self.enhance_contrast(image)

        return image