import cv2
import numpy as np
from keras.models import load_model


model = load_model(r'C:\Users\Admin\PycharmProjects\pythonProject5\model\model.h5')


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    return thresh


def detect_text_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        angles.append(angle)
    return np.mean(angles)


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image


def recognize_text_with_model(image):

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image.reshape(1, processed_image.shape[0], processed_image.shape[1], 1))

    detected_text = ''.join(prediction)
    return detected_text


image_path = 'img.png'
image = cv2.imread(image_path)

# Предварительная обработка изображения
processed_image = preprocess_image(image)

# Определение угла наклона текста
text_skew_angle = detect_text_skew(image)

# Поворот изображения на найденный угол
rotated_image = rotate_image(image, text_skew_angle)

# Распознавание текста с использованием модели
detected_text = recognize_text_with_model(rotated_image)

print("Detected Text:", detected_text)

