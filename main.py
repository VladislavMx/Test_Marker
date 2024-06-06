import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = image.copy()
line_count = 0
for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)

    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text = pytesseract.image_to_string(gray[y:y + h, x:x + w])


    if text.strip() != '':
        line_count += 1
        print(f"Текст строки {line_count}: {text.strip()}")


cv2.imshow('временноефывафыва', output)
cv2.waitKey(0)
cv2.destroyAllWindows()


print(f"количество строк: {line_count}")

print("координаты:")

