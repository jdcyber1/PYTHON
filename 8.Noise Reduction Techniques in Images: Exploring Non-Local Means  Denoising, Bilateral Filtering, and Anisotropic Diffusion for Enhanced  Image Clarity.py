import cv2
import numpy as np
import matplotlib.pyplot as plt


def add_noise(image, noise_type='gaussian'):
    if noise_type == 'gaussian':
        mean = 0
        sigma = 25
        noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    elif noise_type == 'salt_and_pepper':
        noisy_image = image.copy()
        s_vs_p = 0.5
        amount = 0.04

        num_salt = int(np.ceil(amount * image.size * s_vs_p))
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1]] = 255

        num_pepper = int(np.ceil(amount * image.size * (1 - s_vs_p)))
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1]] = 0

        return noisy_image


def non_local_means_denoising(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def bilateral_filter(image):
    return cv2.bilateralFilter(image, d=15, sigmaColor=75, sigmaSpace=75)


def anisotropic_diffusion(image, n_iter=10, kappa=50, gamma=0.1):
    img = image.astype(np.float32)

    for _ in range(n_iter):
        north = np.roll(img, -1, axis=0)
        south = np.roll(img, 1, axis=0)
        east = np.roll(img, -1, axis=1)
        west = np.roll(img, 1, axis=1)

        diff_n = img - north
        diff_s = img - south
        diff_e = img - east
        diff_w = img - west

        c_n = np.exp(-(diff_n / kappa) ** 2)
        c_s = np.exp(-(diff_s / kappa) ** 2)
        c_e = np.exp(-(diff_e / kappa) ** 2)
        c_w = np.exp(-(diff_w / kappa) ** 2)

        img += gamma * (c_n * diff_n + c_s * diff_s + c_e * diff_e + c_w * diff_w)

    return np.clip(img, 0, 255).astype(np.uint8)


def plot_images(original, noisy, nl_denoised, bilateral_denoised, anisotropic_denoised):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Noisy Image")
    plt.imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Non-Local Means Denoised")
    plt.imshow(cv2.cvtColor(nl_denoised, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Bilateral Filter Denoised")
    plt.imshow(cv2.cvtColor(bilateral_denoised, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.figure(figsize=(6, 6))
    plt.title("Anisotropic Diffusion Denoised")
    plt.imshow(cv2.cvtColor(anisotropic_denoised, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()


def main():
    image_path = "test1.jpg"
    original_image = cv2.imread(image_path)

    noisy_image = add_noise(original_image, noise_type='gaussian')
    nl_denoised_image = non_local_means_denoising(noisy_image)
    bilateral_denoised_image = bilateral_filter(noisy_image)
    anisotropic_denoised_image = anisotropic_diffusion(noisy_image)

    plot_images(
        original_image,
        noisy_image,
        nl_denoised_image,
        bilateral_denoised_image,
        anisotropic_denoised_image
    )


if __name__ == "__main__":
    main()
