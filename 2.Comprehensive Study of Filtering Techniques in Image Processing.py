import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path, resize_dim=(512, 512)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, resize_dim)
    return image_resized


def normalize_image(image):
    return image / 255.0


def gaussian_filter(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)


def sobel_filter(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return np.uint8(sobel_magnitude)


def laplacian_filter(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return np.uint8(np.absolute(laplacian))


def plot_comparisons(original, gaussian, median, sobel, laplacian):
    plt.figure(figsize=(14, 10))

    plt.subplot(3, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image (Preprocessed)')
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.imshow(gaussian, cmap='gray')
    plt.title('Gaussian Filter (Noise Reduction)')
    plt.axis('off')

    plt.subplot(3, 2, 3)
    plt.imshow(median, cmap='gray')
    plt.title('Median Filter (Noise Reduction)')
    plt.axis('off')

    plt.subplot(3, 2, 4)
    plt.imshow(sobel, cmap='gray')
    plt.title('Sobel Filter (Edge Detection)')
    plt.axis('off')

    plt.subplot(3, 2, 5)
    plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian Filter (Edge Detection)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def process_image(image_path):
    image = load_image(image_path)
    image_normalized = normalize_image(image)
    image_preprocessed = np.uint8(image_normalized * 255)

    gaussian_image = gaussian_filter(image_preprocessed)
    median_image = median_filter(image_preprocessed)
    sobel_image = sobel_filter(image_preprocessed)
    laplacian_image = laplacian_filter(image_preprocessed)

    plot_comparisons(
        image_preprocessed,
        gaussian_image,
        median_image,
        sobel_image,
        laplacian_image
    )


if __name__ == "__main__":
    process_image("test1.jpg")
