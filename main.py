'''
Based on:
https://laptrinhx.com/ocr-handwriting-recognition-with-opencv-keras-and-tensorflow-2053708178/
'''
from tensorflow.keras.models import load_model
# from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import csv


def main():
    # # Load model
    print("[INFO] loading handwriting OCR model...")
    model = load_model("hiragana_detector.h5")  

    # Load input image and blur for noise reduction
    image_path = "./Images/test2.jpg"
    image = cv2.imread(image_path)
    cv2.namedWindow("Image",2)
    roi = cv2.selectROI("Image", image, False, False)
    cv2.destroyWindow("Image")

    imCrop = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    gray = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (5, 5), 0)
    binarized = cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    (tH, tW) = binarized.shape

    if tW > tH:
	    thresh = imutils.resize(binarized, width = 32)
    else:
        thresh = imutils.resize(binarized, height = 32)

    (tH, tW) = thresh.shape
    dX = int(max(0, 32 - tW) / 2.0)
    dY = int(max(0, 32 - tH) / 2.0)

    padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    padded = cv2.resize(padded, (32, 32))
    padded = padded.astype("float32") / 255.0
    padded = np.expand_dims(padded, axis=-1)

    cv2.namedWindow("Thres32", 2)
    cv2.imshow("Thres32", padded)
    cv2.waitKey()

    symbols = np.array([padded], dtype="float32")

    prediction = model.predict(symbols)

    # List of label names
    CSV_FILE = 'hiragana_classmap.csv'
    with open(CSV_FILE, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        hiragana_dict = {int(rows[0]):chr(int(rows[1], 16)) for rows in reader}

    class_id = np.argmax(prediction)
    prob = prediction[0][class_id]
    label = hiragana_dict[class_id]

    print("[INFO] {} - {:.2f}%".format(label, prob * 100))

    


if __name__ == "__main__":
    main()


# def main():
#     # Load model
#     print("[INFO] loading handwriting OCR model...")
#     model = load_model("hiragana_detector.h5")
    
#     # Load input image and blur for noise reduction
#     image_path = "./Images/s_test_3.jpg"
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Edge detection
#     edged = cv2.Canny(blurred, 30, 150)
#     cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     cnts = sort_contours(cnts, method="left-to-right")[0]
#     chars = []
#     # show the image
#     cv2.imshow("Image", edged)
#     cv2.waitKey(0)

#     # loop over the contours
#     for c in cnts:
#         # compute the bounding box of the contour
#         (x, y, w, h) = cv2.boundingRect(c)

#         # filter out bounding boxes, ensuring they are neither too small
#         # nor too large
#         if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
#             # extract the character and threshold it to make the character
#             # appear as *white* (foreground) on a *black* background, then
#             # grab the width and height of the thresholded image
#             roi = gray[y:y + h, x:x + w]
#             thresh = cv2.threshold(roi, 0, 255,
#                 cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#             (tH, tW) = thresh.shape

#             # if the width is greater than the height, resize along the
#             # width dimension
#             if tW > tH:
#                 thresh = imutils.resize(thresh, width=32)

#             # otherwise, resize along the height
#             else:
#                 thresh = imutils.resize(thresh, height=32)

#             # re-grab the image dimensions (now that its been resized)
#             # and then determine how much we need to pad the width and
#             # height such that our image will be 32x32
#             (tH, tW) = thresh.shape
#             dX = int(max(0, 32 - tW) / 2.0)
#             dY = int(max(0, 32 - tH) / 2.0)

#             # pad the image and force 32x32 dimensions
#             padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
#                 left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
#                 value=(0, 0, 0))
#             padded = cv2.resize(padded, (32, 32))

#             # prepare the padded image for classification via our
#             # handwriting OCR model
#             padded = padded.astype("float32") / 255.0
#             padded = np.expand_dims(padded, axis=-1)

#             # update our list of characters that will be OCR'd
#             chars.append((padded, (x, y, w, h)))

#     # extract the bounding box locations and padded characters
#     boxes = [b[1] for b in chars]
#     chars = np.array([c[0] for c in chars], dtype="float32")

#     preds = model.predict(chars)

#     # List of label names
#     CSV_FILE = 'hiragana_classmap.csv'
#     with open(CSV_FILE, mode='r') as csvfile:
#         reader = csv.reader(csvfile)
#         hiragana_dict = {int(rows[0]):chr(int(rows[1], 16)) for rows in reader}

#     # loop over the predictions and bounding box locations together
#     for (pred, (x, y, w, h)) in zip(preds, boxes):
#         # find the index of the label with the largest corresponding
#         # probability, then extract the probability and label
#         i = np.argmax(pred)
#         prob = pred[i]
#         label = hiragana_dict[i]

#         # draw the prediction on the image
#         print("[INFO] {} - {:.2f}%".format(label, prob * 100))
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(image, label, (x - 10, y - 10),
#             cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)

#     # show the image
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)
