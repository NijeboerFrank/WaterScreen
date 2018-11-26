from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np
from PIL import Image
import os

# Zet op True als je debug info wilt
DEBUG = False
TESTIMAGE = "test_images/test1.jpg"
PRINT_CNTS = False

# Maak een dictionary zodat alle getallen hun weergave hebben.
# 1 betekent segment is aan. 0 betekent segment is uit.
GETALLEN_DICTIONARY = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}


# Schrijf een plaatje naar een bestand
def writeImage(name, image):
    cv2.imwrite(name + ".jpg", image)


# Verwijder de debug files.
def removeDebug():
    os.remove("blurred.jpg")
    os.remove("cijfers_met_rechthoek.jpg")
    os.remove("counter_after_thresh.jpg")
    os.remove("displayCnt.jpg")
    os.remove("edged.jpg")
    os.remove("gray.jpg")
    os.remove("output.jpg")
    try:
        os.remove("roi 1.jpg")
        os.remove("roi 2.jpg")
        os.remove("roi 3.jpg")
        os.remove("roi 4.jpg")
        os.remove("roi 5.jpg")
        os.remove("roi 6.jpg")
    except Exception as e:
        print(e)
    os.remove("thresh.jpg")
    os.remove("warped.jpg")


# Get the screen with the information from the picture
def getScreen(image_location):
    # Maak een zwarte rand om het plaatje en sla het tijdelijk met de naam:
    # 'filename' + _border.jpg
    image_name = add_black_border(image_location)

    # Laad het plaatje
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

    # De variabele die de rechthoek gaat worden waar het schermpje op staat
    displayCnt = None

    img = image.copy()
    # Loop over de contouren
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
            if DEBUG:
                writeImage("displayCnt", img)

    # Centreer het scherm.
    warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    output = four_point_transform(image, displayCnt.reshape(4, 2))

    if DEBUG:
        writeImage("warped", warped)
        writeImage("output", output)

    os.remove(image_name)
    return warped


# Functie die de getallen van het plaatje kan aflezen.
def getNumberFromImage(image_location):
    # Verwijder de oude debug files
    try:
        removeDebug()
    except Exception:
        print("Kon niet alle files vinden om te verwijderen")

    warped = getScreen(image_location)

    # Maak het scherm geblurred zodat we hiermee kunnen werken
    blur = cv2.GaussianBlur(warped, (7, 7), 0)

    # Maak een threshold afbeelding van het plaatje zodat de contouren goed te zien zijn.
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 91, 2)

    if DEBUG:
        writeImage("thresh", thresh)

    # Zoek de contouren in het schermpje
    edges = cv2.Canny(thresh, 0, 150, 0)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    if DEBUG:
        writeImage("counter_after_thresh", edges)

    # De array met de contouren erin.
    digitCnts = []

    # Loop door de contouren die zijn gemaakt in het schermpje
    for c in cnts:
        # x en y zijn de coordinaten van de linkerbovenhoek en w en
        # h zijn de breedte en hoogte
        (x, y, w, h) = cv2.boundingRect(c)

        # Als de contour een hoogte heeft van tussen de 20 en 100 kan het
        # een getal zijn. Ik check ook of de rechthoek ver genoeg naar rechts ligt zodat
        # we niet door alle rechthoeken heen hoeven die toch geen getallen kunnen zijn.
        if (20 <= h <= 100) and 260 >= x >= 90:

            if DEBUG:
                print(x)
                print(y)

            digitCnts.append(c)

    if DEBUG:
        cijfers = warped.copy()
        for d in digitCnts:
            x, y, w, h = cv2.boundingRect(d)
            cijfers = cv2.rectangle(cijfers, (x, y), (x + w, y + h), (0, 255, 0), 2)
        writeImage("cijfers_met_rechthoek", cijfers)

    # Sorteer de contouren van links naar rechts.
    digitCnts = contours.sort_contours(digitCnts,
                                       method="left-to-right")[0]
    if DEBUG:
        print("DigitCnts length is %s" % (len(digitCnts)))
    digits = []

    # Dit wordt gebruikt voor 1, 0 en 7 omdat die soms in 2
    # rechthoeken worden opgedeeld.
    previous_half = False
    large = False
    roi_number = 1
    for c in digitCnts:
        # Coordinaten van een rechthoek weer.
        (x, y, w, h) = cv2.boundingRect(c)
        if not previous_half:
            # Als de rechthoek niet groot genoeg is voor een heel getal
            if w < 20 or h < 50:
                if w < 20 and h > 50:
                    digits.append(1)
                    print("It's a one that fits in one rectangle")
                    continue
                else:
                    if DEBUG:
                        print("It is a one or a zero or a seven")
                    if w > 20:
                        if DEBUG:
                            print("probably seven")
                        large = True

                    previous_half = True
                    continue
            # Ingezoomd beeld van de rechthoek met het getal
            roi = thresh[y:y + h, x:x + w]

            if DEBUG:
                writeImage("roi %s" % roi_number, roi)
                roi_number += 1

            # Verdeel het rechthoekje in 7 segmenten die terugkomen in GETALLEN_DICTIONARY
            # Hier de grootte van de rechthoeken
            (roiH, roiW) = roi.shape
            (dW, dH) = (int(roiW * 0.20), int(roiH * 0.15))
            dHC = int(roiH * 0.05)

            # Hier worden echt de segmenten gemaakt.
            segments = [
                ((4, 0), (w, dH)),  # top
                ((5, 0), (dW + 5, h // 2)),  # top-left
                ((w - dW, 0), (w, h // 2)),  # top-right
                ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
                ((0, h // 2), (dW, h)),  # bottom-left
                ((w - dW - 5, h // 2), (w - 5, h)),  # bottom-right
                ((0, h - dH), (w, h))  # bottom
            ]

            # Maak een array met allemaal nullen
            on = [0] * len(segments)
            wholeThing = cv2.countNonZero(roi)
            if wholeThing / float(w * h) > 0.45:
                continue

            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                # Pak het ingezoomde fragment
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)

                # Als het aantal witte pixels groter is dan de helft
                # geef dan waarde 1 aan dit segment.
                if total / float(area) > 0.40:
                    on[i] = 1
                # Voor linksboven gaat het iets anders door de vorm van de getallen
                elif xA == 5 and total / float(area) > 0.45:
                    on[i] = 1

            try:
                # Zoek naar een getal dat past bij de segment 'codering'
                digit = GETALLEN_DICTIONARY[tuple(on)]
                digits.append(digit)
                if DEBUG:
                    print("Digit is: %s" % (digit))
            except Exception as e:
                print("kon deze niet herkennen: %s" % e)

        # Checks voor de 'gekke' getallen
        elif previous_half and h < 50 and w < 20:
            if large:
                digits.append(7)
                if DEBUG:
                    print("It is a seven")
            else:
                digits.append(1)
                if DEBUG:
                    print("It is a one")
            previous_half = False
            continue
        elif previous_half and h < 50:
            if DEBUG:
                print("it must be zero")
            digits.append(0)
            previous_half = False

    # Return het getal dat op het display staat
    return (magic(digits))


# Method om een array van integers om te zetten in 1 integer
def magic(numbers):
    return int(''.join(["%d" % x for x in numbers]))


# Plak een plaatje op een zwart plaatje
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


# Functie om linkerbovenhoek af te lezen
def readTopLeft(image_location):
    # Verwijder de oude debug files
    try:
        removeDebug()
    except Exception:
        print("Kon niet alle files vinden om te verwijderen")

    warped = getScreen(image_location)

    # TODO vanaf hier verder

# Vergemakkelijkt een snelle test.
def testImage():
    print("Solution is: %s" % (getNumberFromImage(TESTIMAGE)))


try:
    removeDebug()
except Exception:
    None

if __name__ == "__main__":
    testImage()
