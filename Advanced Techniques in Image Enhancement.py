import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = ((image - min_val) / (max_val - min_val)) * 255
    return np.uint8(stretched)

def adaptive_histogram_equalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def plot_comparisons(original, hist_eq, contrast_stretch, ahe):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(hist_eq, cmap='gray')
    plt.title('Histogram Equalization')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(contrast_stretch, cmap='gray')
    plt.title('Contrast Stretching')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(ahe, cmap='gray')
    plt.title('Adaptive Histogram Equalization')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def process_image(image_path):
    image = load_image(image_path)
    if image is None:
        return
    hist_eq_image = histogram_equalization(image)
    contrast_stretch_image = contrast_stretching(image)
    ahe_image = adaptive_histogram_equalization(image)
    plot_comparisons(image, hist_eq_image, contrast_stretch_image, ahe_image)

if __name__ == "__main__":
    process_image("test1.jpg")
