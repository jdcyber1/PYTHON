import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydicom


def load_dicom_image(dicom_path):
    dicom_data = pydicom.dcmread(dicom_path)
    image = dicom_data.pixel_array
    return image, dicom_data


def enhance_image(image):
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = image.astype(np.uint8)
    enhanced = cv2.GaussianBlur(image, (5, 5), 0)
    return enhanced


def region_growing_segmentation(image, seed_point, threshold):
    height, width = image.shape
    segmented_image = np.zeros_like(image, dtype=np.uint8)

    seed_x, seed_y = seed_point
    seed_value = image[seed_x, seed_y]

    pixel_list = [(seed_x, seed_y)]

    while pixel_list:
        x, y = pixel_list.pop(0)

        if segmented_image[x, y] == 255:
            continue

        if abs(int(image[x, y]) - int(seed_value)) <= threshold:
            segmented_image[x, y] = 255

            if x > 0:
                pixel_list.append((x - 1, y))
            if x < height - 1:
                pixel_list.append((x + 1, y))
            if y > 0:
                pixel_list.append((x, y - 1))
            if y < width - 1:
                pixel_list.append((x, y + 1))

    return segmented_image


def plot_images(images, titles):
    plt.figure(figsize=(15, 6))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    dicom_path = "0009.dcm"
    image, _ = load_dicom_image(dicom_path)

    if len(image.shape) == 3:
        image = image[image.shape[0] // 2]

    enhanced_image = enhance_image(image)

    seed_point = (100, 100)
    threshold = 15

    segmented_image = region_growing_segmentation(
        enhanced_image, seed_point, threshold
    )

    plot_images(
        [image, enhanced_image, segmented_image],
        ["Original DICOM Slice", "Enhanced Image", "Segmented Image"]
    )


if __name__ == "__main__":
    main()
