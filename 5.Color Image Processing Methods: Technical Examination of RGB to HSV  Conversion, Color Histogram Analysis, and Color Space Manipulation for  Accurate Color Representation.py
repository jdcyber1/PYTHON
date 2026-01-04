import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (400, 400))
    return image_rgb


def rgb_to_hsv(image_rgb):
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)


def rgb_to_lab(image_rgb):
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)


def plot_images(images, titles):
    plt.figure(figsize=(15, 6))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def color_histogram_rgb(image_rgb):
    colors = ('r', 'g', 'b')
    plt.figure()
    for i, color in enumerate(colors):
        histogram = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        plt.plot(histogram, color=color)
        plt.xlim([0, 256])
    plt.title("RGB Color Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()


def color_histogram_hsv(image_hsv):
    plt.figure()

    hue_hist = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])
    sat_hist = cv2.calcHist([image_hsv], [1], None, [256], [0, 256])
    val_hist = cv2.calcHist([image_hsv], [2], None, [256], [0, 256])

    plt.subplot(3, 1, 1)
    plt.plot(hue_hist, color='r')
    plt.title("Hue Histogram")
    plt.xlim([0, 180])

    plt.subplot(3, 1, 2)
    plt.plot(sat_hist, color='g')
    plt.title("Saturation Histogram")
    plt.xlim([0, 256])

    plt.subplot(3, 1, 3)
    plt.plot(val_hist, color='b')
    plt.title("Value Histogram")
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()


def split_rgb_channels(image_rgb):
    r_channel = image_rgb[:, :, 0]
    g_channel = image_rgb[:, :, 1]
    b_channel = image_rgb[:, :, 2]
    return r_channel, g_channel, b_channel


def lab_brightness_adjustment(image_rgb, adjustment=20):
    lab_image = rgb_to_lab(image_rgb)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    l_channel = np.clip(l_channel + adjustment, 0, 255).astype(np.uint8)

    adjusted_lab = cv2.merge((l_channel, a_channel, b_channel))
    adjusted_rgb = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2RGB)
    return adjusted_rgb


def color_histogram_lab(image_lab):
    plt.figure()

    l_hist = cv2.calcHist([image_lab], [0], None, [256], [0, 256])
    a_hist = cv2.calcHist([image_lab], [1], None, [256], [0, 256])
    b_hist = cv2.calcHist([image_lab], [2], None, [256], [0, 256])

    plt.subplot(3, 1, 1)
    plt.plot(l_hist, color='black')
    plt.title("L Channel Histogram")
    plt.xlim([0, 256])

    plt.subplot(3, 1, 2)
    plt.plot(a_hist, color='green')
    plt.title("A Channel Histogram")
    plt.xlim([0, 256])

    plt.subplot(3, 1, 3)
    plt.plot(b_hist, color='blue')
    plt.title("B Channel Histogram")
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()


def color_image_processing(image_path):
    image_rgb = load_image(image_path)

    hsv_image = rgb_to_hsv(image_rgb)
    lab_image = rgb_to_lab(image_rgb)

    color_histogram_rgb(image_rgb)
    color_histogram_hsv(hsv_image)
    color_histogram_lab(lab_image)

    r_channel, g_channel, b_channel = split_rgb_channels(image_rgb)
    plot_images(
        [r_channel, g_channel, b_channel],
        ["Red Channel", "Green Channel", "Blue Channel"]
    )

    brightened_image = lab_brightness_adjustment(image_rgb, adjustment=40)
    plot_images(
        [image_rgb, brightened_image],
        ["Original Image", "Brightness Adjusted Image"]
    )

    plot_images(
        [image_rgb, hsv_image, lab_image],
        ["RGB Image", "HSV Image", "LAB Image"]
    )


if __name__ == "__main__":
    color_image_processing("test2.jpg")
