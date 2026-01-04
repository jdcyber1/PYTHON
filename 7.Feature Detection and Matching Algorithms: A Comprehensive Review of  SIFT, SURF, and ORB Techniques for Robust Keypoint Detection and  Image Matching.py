import cv2
import numpy as np
import matplotlib.pyplot as plt
import zlib
import os


def load_image(image_path):
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, grayscale_image


def display_image(image, title):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


def jpeg_compression(image_path, quality):
    image = cv2.imread(image_path)
    compressed_image_path = "compressed_image.jpg"

    cv2.imwrite(
        compressed_image_path,
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )

    original_size = os.path.getsize(image_path)
    compressed_size = os.path.getsize(compressed_image_path)

    print(f"Original size: {original_size / 1024:.2f} KB")
    print(f"Compressed size (JPEG, quality={quality}): {compressed_size / 1024:.2f} KB")

    compressed_image = cv2.imread(compressed_image_path)
    display_image(compressed_image, f"JPEG Compression (Quality={quality})")


def png_compression(image_path, compression_level):
    image = cv2.imread(image_path)
    compressed_image_path = "compressed_image.png"

    cv2.imwrite(
        compressed_image_path,
        image,
        [int(cv2.IMWRITE_PNG_COMPRESSION), compression_level]
    )

    original_size = os.path.getsize(image_path)
    compressed_size = os.path.getsize(compressed_image_path)

    print(f"Original size: {original_size / 1024:.2f} KB")
    print(
        f"Compressed size (PNG, level={compression_level}): "
        f"{compressed_size / 1024:.2f} KB"
    )

    compressed_image = cv2.imread(compressed_image_path)
    display_image(compressed_image, f"PNG Compression (Level={compression_level})")


def run_length_encoding(image):
    pixels = image.flatten()
    compressed_data = []
    count = 1

    for i in range(1, len(pixels)):
        if pixels[i] == pixels[i - 1]:
            count += 1
        else:
            compressed_data.append((pixels[i - 1], count))
            count = 1

    compressed_data.append((pixels[-1], count))
    return compressed_data


def rle_compression(image_path):
    _, grayscale_image = load_image(image_path)

    rle_data = run_length_encoding(grayscale_image)
    original_size = grayscale_image.size
    compressed_size = len(rle_data)

    print(f"Original size (pixels): {original_size}")
    print(f"Compressed size (RLE): {compressed_size}")
    print(f"First 10 runs: {rle_data[:10]}")


def zlib_compression(image_path):
    _, grayscale_image = load_image(image_path)

    original_size = grayscale_image.nbytes
    compressed_data = zlib.compress(grayscale_image.tobytes())
    compressed_size = len(compressed_data)

    print(f"Original size (bytes): {original_size}")
    print(f"Compressed size (zlib): {compressed_size}")

    decompressed_data = zlib.decompress(compressed_data)
    decompressed_image = np.frombuffer(
        decompressed_data,
        dtype=np.uint8
    ).reshape(grayscale_image.shape)

    plt.imshow(decompressed_image, cmap='gray')
    plt.title("Decompressed Image (zlib)")
    plt.axis('off')
    plt.show()


def image_compression_demo(image_path):
    image, _ = load_image(image_path)
    display_image(image, "Original Image")

    jpeg_compression(image_path, quality=50)
    png_compression(image_path, compression_level=9)

    print("\nRun-Length Encoding Compression:")
    rle_compression(image_path)

    print("\nLossless Compression (zlib):")
    zlib_compression(image_path)


if __name__ == "__main__":
    image_compression_demo("test1.jpg")

