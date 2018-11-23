from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np
from PIL import Image
import os

# Zet op True als je debug info wilt
DEBUG = True
TESTIMAGE = "test_images/181116-015510.jpg"
PRINT_CNTS = False

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

def firstSteps(image_location):
    # Maak een zwarte rand om het plaatje en sla het tijdelijk met de naam:
    # 'filename' + _border.jpg
    image_name = add_black_border(image_location)

    # laadt het plaatje
    image = cv2.imread(image_name)

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
    if DEBUG:
        writeImage("warped", warped)
        writeImage("output", output)

    blur = cv2.GaussianBlur(warped, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 91, 2)

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
        if w >= 0 and (h >= 20 and h <= 100) and 260 >= x >= 130:
            print(x)
            print(y)
            digitCnts.append(c)

    if DEBUG:
        cijfers = output.copy()
        for d in digitCnts:
            x, y, w, h = cv2.boundingRect(d)
            cijfers = cv2.rectangle(cijfers, (x, y), (x + w, y + h), (0, 255, 0), 2)
        writeImage("cijfers_met_rechthoek", cijfers)

    os.remove(image_name)

    # Sorteer de contouren van links naar rechts.
    digitCnts = contours.sort_contours(digitCnts,
                                       method="left-to-right")[0]
    digits = []


    for c in digitCnts:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y:y + h, x:x + w]

        # compute the width and height of each of the 7 segments
        # we are going to examine
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
        dHC = int(roiH * 0.05)

        # define the set of 7 segments
        segments = [
            ((0, 0), (w, dH)),  # top
            ((0, 0), (dW, h // 2)),  # top-left
            ((w - dW, 0), (w, h // 2)),  # top-right
            ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
            ((0, h // 2), (dW, h)),  # bottom-left
            ((w - dW, h // 2), (w, h)),  # bottom-right
            ((0, h - dH), (w, h))  # bottom
        ]
        on = [0] * len(segments)

    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        # extract the segment ROI, count the total number of
        # thresholded pixels in the segment, and then compute
        # the area of the segment
        segROI = roi[yA:yB, xA:xB]
        total = cv2.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)

        # if the total number of non-zero pixels is greater than
        # 50% of the area, mark the segment as "on"
        if total / float(area) > 0.5:
            on[i] = 1

        # lookup the digit and draw it on the image
    digit = GETALLEN_DICTIONARY[tuple(on)]
    digits.append(digit)
    for digi in digits:
        print(digi)


def add_black_border(image):
    name = os.path.splitext(image)[0]
    old_im = Image.open(image)
    old_size = old_im.size

    new_size = (500, 500)
    new_im = Image.new("RGB", new_size)  ## luckily, this is already black!
    new_im.paste(old_im, (int((new_size[0] - old_size[0]) / 2),
                          int((new_size[1] - old_size[1]) / 2)))
    new_im.save(name + "_border.jpg")
    return name + "_border.jpg"

def testImage():
    firstSteps(TESTIMAGE)


if DEBUG:
    testImage()