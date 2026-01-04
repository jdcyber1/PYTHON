AIM
To detect and match features between two images using SIFT, SURF, and ORB algorithms, 
visualize the key points, and display the feature matches.
ALGORITHM
Step 1: Load Images
Load two images for feature detection and matching.
Step 2: SIFT Detection:
i) Detect keypoints and compute descriptors using the SIFT algorithm
ii) Display keypoints on both images.
iii) Match features between the two images and display the matches.
Step 3: SURF Detection (optional, commented out in code):
i) Detect keypoints and compute descriptors using the SURF algorithm.
ii) Display keypoints on both images.
iii) Match features and display the matches.
Step 4: ORB Detection
i)Detect keypoints and compute descriptors using the ORB algorithm.
ii) Display keypoints on both images.
iii) Match features and display the matches.
Step 5: Display Results
For each feature detection algorithm, display the key points and matches between the 
two images.
CODING
import cv2
import numpy as np
import matplotlib.pyplot as plt
def load_images(img1_path, img2_path): 
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path) 
return img1, img2
def display_keypoints(image, keypoints, title):
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, 
flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)) 
plt.title(title)
plt.axis('off') 
plt.show()
def match_features(des1, des2, method):
bf = cv2.BFMatcher(cv2.NORM_L2 if method != 'ORB' else cv2.NORM_HAMMING, 
crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance) 
return matches
def display_matches(img1, img2, kp1, kp2, matches, title):
matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2) 
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.title(title) 
plt.axis('off') 
plt.show()
defsift_detection(img1, img2): 
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None) 
kp2, des2 = sift.detectAndCompute(img2, None)
display_keypoints(img1, kp1, "SIFT Keypoints - Image 1") 
display_keypoints(img2, kp2, "SIFT Keypoints - Image 2")
matches = match_features(des1, des2, method='SIFT') 
display_matches(img1, img2, kp1, kp2, matches, "SIFT Feature Matching")
defsurf_detection(img1, img2):
surf = cv2.xfeatures2d.SURF_create(400)
kp1, des1 = surf.detectAndCompute(img1, None) 
kp2, des2 = surf.detectAndCompute(img2, None)
display_keypoints(img1, kp1, "SURF Keypoints - Image 1") 
display_keypoints(img2, kp2, "SURF Keypoints - Image 2")
matches = match_features(des1, des2, method='SURF') 
display_matches(img1, img2, kp1, kp2, matches, "SURF Feature Matching")
def orb_detection(img1, img2): 
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None) 
kp2, des2 = orb.detectAndCompute(img2, None)
display_keypoints(img1, kp1, "ORB Keypoints - Image 1") 
display_keypoints(img2, kp2, "ORB Keypoints - Image 2")
matches = match_features(des1, des2, method='ORB') display_matches(img1,
img2, kp1, kp2, matches, "ORB Feature Matching")
def feature_detection_demo(img1_path, img2_path): img1, img2 
= load_images(img1_path, img2_path)
print("SIFT Feature Detection and Matching...") sift_detection(img1,
img2)
# print("SURF Feature Detection and Matching...") # 
surf_detection(img1, img2)
print("ORB Feature Detection and Matching...") 
orb_detection(img1, img2)
img1_path = 'test1.jpg' 
img2_path = 'test2.jpg'
feature_detection_demo(img1_path, img2_path)
