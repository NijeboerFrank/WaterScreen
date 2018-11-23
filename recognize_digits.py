from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np

# Zet op True als je debug info wilt
DEBUG = True
TESTIMAGE = "test_images/test.jpg"

# Maak een dictionary zodat alle getallen hun weergave hebben.
# 1 betekent segment is aan. 0 betekent segment is uit.
GETALLEN_DICTIONARY = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

def writeImage(name, image):
    cv2.imwrite(name + ".jpg",image)

def firstSteps(imagelocation):
    # laad het plaatje
    image = cv2.imread(imagelocation)

    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)
    if DEBUG:
        writeImage("gray", gray)
        writeImage("edged", edged)
        writeImage("blurred", blurred)

def testImage():
    firstSteps(TESTIMAGE)


if DEBUG:
    testImage()