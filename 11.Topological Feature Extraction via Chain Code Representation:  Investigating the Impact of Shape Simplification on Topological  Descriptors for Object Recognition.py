import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to read the image.")
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary_image


def extract_contours(binary_image):
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def compute_chain_code(contour):
    chain_code = []
    directions = {
        (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7
    }

    for i in range(len(contour)):
        p1 = contour[i][0]
        p2 = contour[(i + 1) % len(contour)][0]
        dx = np.sign(p2[0] - p1[0])
        dy = np.sign(p2[1] - p1[1])
        chain_code.append(directions.get((dx, dy), 0))

    return chain_code


def simplify_chain_code(chain_code):
    simplified_code = []
    for code in chain_code:
        if not simplified_code or simplified_code[-1] != code:
            simplified_code.append(code)
    return simplified_code


def plot_results(original_contour, simplified_contour, chain_code, simplified_code):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(
        original_contour[:, 0, 0],
        original_contour[:, 0, 1],
        "b-",
        label="Original Contour",
    )
    plt.title("Original Contour")
    plt.axis("equal")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(
        simplified_contour[:, 0, 0],
        simplified_contour[:, 0, 1],
        "r-",
        label="Simplified Contour",
    )
    plt.title("Simplified Contour")
    plt.axis("equal")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.text(
        0.5,
        0.5,
        f"Original Chain Code:\n{chain_code}",
        ha="center",
        va="center",
        fontsize=10,
    )
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.text(
        0.5,
        0.5,
        f"Simplified Chain Code:\n{simplified_code}",
        ha="center",
        va="center",
        fontsize=10,
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main(image_path):
    binary_image = preprocess_image(image_path)
    contours = extract_contours(binary_image)

    if len(contours) == 0:
        print("No contours found!")
        return

    original_contour = contours[0]
    chain_code = compute_chain_code(original_contour)
    simplified_code = simplify_chain_code(chain_code)

    epsilon = 5
    simplified_contour = cv2.approxPolyDP(original_contour, epsilon, True)

    plot_results(original_contour, simplified_contour, chain_code, simplified_code)


if __name__ == "__main__":
    image_path = "test1.jpg"
    main(image_path)
