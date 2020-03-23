from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

#take command line args
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# define correct answer
ANSWER_KEY = {  0: 1, 
                1: 4, 
                2: 0, 
                3: 3, 
                4: 1    }

image = cv2.imread(args["image"])  # read image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grey scale image

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # otsu thresholding

# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)  # get dimension of contours
    ar = w / float(h)

    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1: # find valid question contours
        questionCnts.append(c)

# sort the question contours top-to-bottom
questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
correct = 0

# traversing the questions
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
	cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
	bubbled = None
    
    # loop over the sorted contours
	for (j, c) in enumerate(cnts):
		# construct a mask that reveals only the current "bubble" for the question
		mask = np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
		# apply the mask to the thresholded image, then count the number of non-zero pixels in the bubble area
		mask = cv2.bitwise_and(thresh, thresh, mask=mask)
		total = cv2.countNonZero(mask)
		# if the current total has a larger number of total non-zero pixels, then we are examining the currently bubbled-in answer
		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)

    # initialize the contour color and the index of the correct answer
	color = (0, 0, 255)
	k = ANSWER_KEY[q]
	# check to see if the bubbled answer is correct
	if k == bubbled[1]:
		color = (0, 255, 0)
		correct += 1
	# draw the outline of the correct answer on the test
	cv2.drawContours(image, [cnts[k]], -1, color, 3)

# grab the test taker
score = (correct / 5.0) * 100
print("score: {:.2f}%".format(score))
cv2.putText(image, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)


cv2.imshow("OMR", image)
    
cv2.waitKey(0)
