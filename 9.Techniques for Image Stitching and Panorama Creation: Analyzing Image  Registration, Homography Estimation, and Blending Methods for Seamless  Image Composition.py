import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_and_compute_features(img):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


def match_features(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def find_homography(kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return homography


def stitch_images(img1, img2):
    kp1, des1 = detect_and_compute_features(img1)
    kp2, des2 = detect_and_compute_features(img2)

    matches = match_features(des1, des2)
    homography = find_homography(kp1, kp2, matches)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    result = cv2.warpPerspective(img2, homography, (w1 + w2, h1))
    result[0:h1, 0:w1] = img1
    return result


def create_panorama(images):
    panorama = images[0]
    for i in range(1, len(images)):
        panorama = stitch_images(panorama, images[i])
    return panorama


def plot_images(images, titles):
    plt.figure(figsize=(15, 8))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    img1 = cv2.imread("1.jpg")
    img2 = cv2.imread("test2.jpg")
    img3 = cv2.imread("test1.jpg")

    images = [img1, img2, img3]
    panorama = create_panorama(images)

    plot_images(images + [panorama],
                ["Image 1", "Image 2", "Image 3", "Panorama"])

    cv2.imwrite("panorama_result.jpg", panorama)


if __name__ == "__main__":
    main()
