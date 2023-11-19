import cv2
import numpy as np
import os

# Load the image
filename = 'image.jpg'
img = cv2.imread(filename)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the face cascade and detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Load the eye cascade and detect eyes
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load the smile cascade and detect smiles
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Define the colors for the eye color detection
lower_color = np.array([0, 0, 0])
upper_color = np.array([255, 255, 255])

# Define a function to get the average color within a given region
def get_average_color(image, region):
    r, g, b = 0, 0, 0
    count = 0
    for x in range(region[0], region[0]+region[2]):
        for y in range(region[1], region[1]+region[3]):
            color = image[y, x]
            r += color[2]
            g += color[1]
            b += color[0]
            count += 1
    return (b//count, g//count, r//count)

# Draw rectangles around the detected faces, eyes, and smiles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        eye_region = roi_color[ey:ey+eh, ex:ex+ew]
        mask = cv2.inRange(eye_region, lower_color, upper_color)
        avg_color = get_average_color(eye_region, (0, 0, ew, eh))
        cv2.putText(roi_color, f"Eye color: {avg_color}", (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
        cv2.putText(roi_color, "Smiling", (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Save the modified image
filename_output = os.path.splitext(filename)[0] + '_output.jpg'
cv2.imwrite(filename_output, img)

# Display the image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
import cv2

# Load the image
img = cv2.imread('image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the face cascade and detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Load the eye cascade and detect eyes
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load the smile cascade and detect smiles
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Draw rectangles around the detected faces, eyes, and smiles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

# Display the image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""