import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path, resize_dim=(512, 512)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    image_resized = cv2.resize(binary_image, resize_dim)
    return image_resized


def plot_images(original, result, operation_name):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(result, cmap='gray')
    plt.title(f'{operation_name} Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def dilation(image, kernel_size=(5, 5), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)


def erosion(image, kernel_size=(5, 5), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)


def opening(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def closing(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def skeletonization(image):
    skeleton = np.zeros(image.shape, np.uint8)
    temp_img = image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(temp_img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(temp_img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        temp_img = eroded.copy()

        if cv2.countNonZero(temp_img) == 0:
            break

    return skeleton


def process_morphological_operations(image_path):
    image = load_image(image_path)

    plot_images(image, dilation(image), 'Dilation')
    plot_images(image, erosion(image), 'Erosion')
    plot_images(image, opening(image), 'Opening')
    plot_images(image, closing(image), 'Closing')
    plot_images(image, skeletonization(image), 'Skeletonization')


if __name__ == "__main__":
    process_morphological_operations("test2.jpg")
