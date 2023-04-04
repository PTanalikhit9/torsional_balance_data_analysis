
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

def analyze_images(image1_path, image2_path):
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    images = [image1, image2]
    titles = ['Before', 'After']

    for i, (image, title) in enumerate(zip(images, titles)):
        x = np.linspace(0, image.shape[1] - 1, image.shape[1])
        y = np.linspace(0, image.shape[0] - 1, image.shape[0])

        # Intensity along X
        intensity_x = image.sum(axis=0)

        # Skewness along X
        skew_x = skew(intensity_x)
        print(f"{title} - Skewness along X axis: {skew_x:.4f}")

        # Intensity along Y
        intensity_y = image.sum(axis=1)

        # Skewness along Y
        skew_y = skew(intensity_y)
        print(f"{title} - Skewness along Y axis: {skew_y:.4f}")

if __name__ == '__main__':
    before_image_path = 'before.PNG'
    after_image_path = 'after.PNG'
    analyze_images(before_image_path, after_image_path)
