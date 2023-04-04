
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, amplitude, mu, sigma, offset):
    return amplitude * np.exp(-((x - mu) / sigma)**2 / 2) + offset

def analyze_images(image1_path, image2_path):
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    images = [image1, image2]
    titles = ['Before', 'After']

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    for i, (image, title) in enumerate(zip(images, titles)):
        x = np.linspace(0, image.shape[1] - 1, image.shape[1])
        y = np.linspace(0, image.shape[0] - 1, image.shape[0])
        X, Y = np.meshgrid(x, y)

        # Contour plot
        axes[0, i].contourf(X, Y, image, cmap='viridis')
        axes[0, i].set_title(f'{title} - Top View Contour Plot')

        # Intensity along X
        intensity_x = image.sum(axis=0)
        axes[1, i].plot(x, intensity_x)
        axes[1, i].set_title(f'{title} - Intensity along X axis')

        # Gaussian fit along X
        initial_guess_x = (intensity_x.max(), x[intensity_x.argmax()], 20, 0)
        popt_x, _ = curve_fit(gaussian, x, intensity_x, p0=initial_guess_x)
        axes[1, i].plot(x, gaussian(x, *popt_x), 'r--', label=f'Amplitude={popt_x[0]:.1f}, Mean={popt_x[1]:.1f}, Sigma={popt_x[2]:.1f}')
        axes[1, i].legend(loc='upper right')

        # Intensity along Y
        intensity_y = image.sum(axis=1)
        axes[2, i].plot(y, intensity_y)
        axes[2, i].set_title(f'{title} - Intensity along Y axis')

        # Gaussian fit along Y
        initial_guess_y = (intensity_y.max(), y[intensity_y.argmax()], 20, 0)
        popt_y, _ = curve_fit(gaussian, y, intensity_y, p0=initial_guess_y)
        axes[2, i].plot(y, gaussian(y, *popt_y), 'r--', label=f'Amplitude={popt_y[0]:.1f}, Mean={popt_y[1]:.1f}, Sigma={popt_y[2]:.1f}')
        axes[2, i].legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    # plt.savefig('analy.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    before_image_path = 'before.PNG'
    after_image_path = 'after.PNG'
    analyze_images(before_image_path, after_image_path)
