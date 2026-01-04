import cv2
import numpy as np
from skimage import feature
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read the image.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = feature.local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-6
    return hist


def extract_edge_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    hist = cv2.calcHist([edges], [0], None, [256], [0, 256]).flatten()
    hist /= hist.sum() + 1e-6
    return hist


def extract_geometric_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros(5)

    contour = contours[0]
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    moments = cv2.moments(contour)

    cx = moments["m10"] / moments["m00"] if moments["m00"] != 0 else 0
    cy = moments["m01"] / moments["m00"] if moments["m00"] != 0 else 0

    return np.array([area, perimeter, cx, cy, len(contours)])


def combine_features(texture, edges, geometry):
    return np.hstack([texture, edges, geometry])


def train_classifier(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    svm = SVC(kernel="linear")
    svm.fit(X_pca, y)

    return svm, scaler, pca


def classify_images(images, svm, scaler, pca):
    predictions = []
    for img in images:
        t = extract_texture_features(img)
        e = extract_edge_features(img)
        g = extract_geometric_features(img)
        features = combine_features(t, e, g)
        features = scaler.transform([features])
        features = pca.transform(features)
        predictions.append(svm.predict(features)[0])
    return predictions


def evaluate_classifier(y_true, y_pred):
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def main(image_paths, labels):
    feature_list = []

    for path in image_paths:
        img = load_image(path)
        t = extract_texture_features(img)
        e = extract_edge_features(img)
        g = extract_geometric_features(img)
        feature_list.append(combine_features(t, e, g))

    X = np.array(feature_list)
    y = np.array(labels)

    svm, scaler, pca = train_classifier(X, y)
    images = [load_image(p) for p in image_paths]
    predictions = classify_images(images, svm, scaler, pca)
    evaluate_classifier(y, predictions)


if __name__ == "__main__":
    image_paths = [
        "test1.jpg",
        "1.jpg",
        "test2.jpg"
    ]
    labels = [0, 1, 0]
    main(image_paths, labels)
