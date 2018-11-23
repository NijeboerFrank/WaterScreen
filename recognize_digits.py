from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np

# Zet op True als je debug info wilt
DEBUG = True
TESTIMAGE = "test_images/test2.jpg"

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
    # Maak foto zwart-wit
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur de foto. Dit is om de noise te verminderen
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Detecteer de contouren in de foto.
    edged = cv2.Canny(blurred, 0, 150, 0)

    if DEBUG:
        writeImage("gray", gray)
        writeImage("edged", edged)
        writeImage("blurred", blurred)

    # Zoek naar contouren in de foto en zet ze in een array
    # Sorteer vervolgens de array van groot naar klein.
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None

    img = image.copy()
    # loop over de contouren
    for c in cnts:
        print("hoi")
        # Benader de contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Als de contour ongeveer 4 hoeken heeft dan moeten we daar op inzoomen.
        # Zo krijgen we altijd het beeld van het schermpje en kan de camera ook eventueel
        # iets gedraaid zijn.
        if len(approx) == 4:
            displayCnt = approx
            x, y, w, h = cv2.boundingRect(displayCnt)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            writeImage("displayCnt", img)


    # Centreer het scherm.
    warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    output = four_point_transform(image, displayCnt.reshape(4, 2))
    blurredwarped = four_point_transform(blurred, displayCnt.reshape(4, 2))
    if DEBUG:
        writeImage("warped", warped)
        writeImage("output", output)

    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

    if DEBUG:
        writeImage("thresh", thresh)

    # find contours in the thresholded image, then initialize the
    # digit contours lists
    edges = cv2.Canny(thresh, 0, 150, 0)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    writeImage("counter_after_thresh", edges)
    digitCnts = []

    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # if the contour is sufficiently large, it must be a digit
        digitCnts.append(c)

    cijfers = output.copy()
    for d in digitCnts:
        print("hallo")
        x, y, w, h = cv2.boundingRect(d)
        cijfers = cv2.rectangle(cijfers, (x, y), (x + w, y + h), (0, 255, 0), 2)
        writeImage("displayCnt", cijfers)

    if DEBUG:
        writeImage("cijfers_met_rechthoek", cijfers)

def testImage():
    firstSteps(TESTIMAGE)


if DEBUG:
    testImage()