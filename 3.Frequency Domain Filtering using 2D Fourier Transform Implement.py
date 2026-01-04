import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path, resize_dim=(512, 512)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, resize_dim)
    return image_resized


def fourier_transform(image):
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)
    return dft_shifted


def create_low_pass_filter(shape, cutoff_radius):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)

    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2) <= cutoff_radius:
                mask[i, j] = 1
    return mask


def create_high_pass_filter(shape, cutoff_radius):
    low_pass = create_low_pass_filter(shape, cutoff_radius)
    return 1 - low_pass


def apply_frequency_filter(dft_shifted, filter_mask):
    return dft_shifted * filter_mask


def inverse_fourier_transform(dft_filtered):
    dft_inverse_shifted = np.fft.ifftshift(dft_filtered)
    img_back = np.fft.ifft2(dft_inverse_shifted)
    img_back = np.abs(img_back)
    return img_back


def plot_images(original, smoothed, sharpened):
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(smoothed, cmap='gray')
    plt.title('Smoothed Image (Low-Pass Filter)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(sharpened, cmap='gray')
    plt.title('Sharpened Image (High-Pass Filter)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def process_image(image_path):
    image = load_image(image_path)

    dft_shifted = fourier_transform(image)

    low_pass_filter = create_low_pass_filter(image.shape, cutoff_radius=50)
    high_pass_filter = create_high_pass_filter(image.shape, cutoff_radius=30)

    dft_low_pass = apply_frequency_filter(dft_shifted, low_pass_filter)
    smoothed_image = inverse_fourier_transform(dft_low_pass)

    dft_high_pass = apply_frequency_filter(dft_shifted, high_pass_filter)
    sharpened_image = inverse_fourier_transform(dft_high_pass)

    plot_images(image, smoothed_image, sharpened_image)


if __name__ == "__main__":
    process_image("test2.jpg")
