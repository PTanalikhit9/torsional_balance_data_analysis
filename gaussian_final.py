import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

def gaussian_2d(x_data_tuple, amplitude, xo, yo, sigma_x, sigma_y, offset):
    (x, y) = x_data_tuple
    xo, yo = float(xo), float(yo)
    g = offset + amplitude * np.exp(-(((x - xo) / sigma_x) ** 2 + ((y - yo) / sigma_y) ** 2) / 2)
    return g.ravel()

def plot_intensity_data_3d(image, ax, color):
    x = np.linspace(0, image.shape[1] - 1, image.shape[1])
    y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, image, color=color, alpha=0.5)

def analyze_images(image1_path, image2_path):
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_intensity_data_3d(image1, ax, 'coral')
    plot_intensity_data_3d(image2, ax, 'skyblue')

    # Dummy scatter plots for legend
    ax.scatter([], [], [], c='coral', marker='o', label='Before')
    ax.scatter([], [], [], c='skyblue', marker='o', label='After')
    ax.legend()

    # ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    plt.show()

    peaks = []
    for image in [image1, image2]:
        x = np.linspace(0, image.shape[1] - 1, image.shape[1])
        y = np.linspace(0, image.shape[0] - 1, image.shape[0])
        X, Y = np.meshgrid(x, y)

        initial_guess = (image.max(), image.shape[1] / 2, image.shape[0] / 2, 20, 20, 0)
        popt, _ = curve_fit(gaussian_2d, (X, Y), image.ravel(), p0=initial_guess)

        peak_x, peak_y = popt[1], popt[2]
        peaks.append((peak_x, peak_y))

    peaks = np.array(peaks)
    plt.scatter(peaks[:, 0], peaks[:, 1], c=['r', 'b'], marker='o')
    plt.plot(peaks[:, 0], peaks[:, 1], 'k--')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Laser Beam Peaks')
    plt.axis('equal')  # Make the scale of x-axis and y-axis the same

    x_min, x_max = 380, 435  # Specify the desired x-axis limits here
    plt.xlim(x_min, x_max)

    for i, peak in enumerate(peaks):
        # plt.text(peak[0], peak[1], f'({peak[0]:.2f}, {peak[1]:.2f})', fontsize=12)
        plt.text(peak[0]-7, peak[1]-4, f'({peak[0]:.2f}, {peak[1]:.2f})', fontsize=12)
    # plt.show()
    # print(peak[1])

    displacement = np.sqrt((peaks[1, 0] - peaks[0, 0])**2 + (peaks[1, 1] - peaks[0, 1])**2)


    # Display displacement at the top left corner of the plot
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    displacement_text = f"Displacement: {displacement:.2f} a.u."
    plt.text(xmin + 2, ymax + 18, displacement_text, fontsize=12, verticalalignment='top')


    # plt.text(xmin + 0.02*(xmax-xmin), ymax - 0.05*(ymax-ymin), displacement_text, fontsize=12, verticalalignment='top')
    # print(f"Displacement: {displacement:.2f} a.u.")

    plt.show()

if __name__ == '__main__':
    before_image_path = 'before.PNG'
    after_image_path = 'after.PNG'
    analyze_images(before_image_path, after_image_path)
