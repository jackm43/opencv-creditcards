# USAGE
# python ocr_template_match.py --image images/credit_card_01.png --reference ocr_a_reference.png

# import the necessary packages
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2 as cv
import os

# directory based implementation
directory = os.fsencode("/Users/jackmunro/Documents/Coding/credit-card-recog/")
ref_image = "/Users/jackmunro/Documents/Coding/credit-card-recog/ocr_a_reference.png"

FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}

# load the reference OCR-A image from disk, convert it to grayscale,
# and threshold it, such that the digits appear as *white* on a
# *black* background
# and invert it, such that the digits appear as *white* on a *black*
ref = cv.imread(ref_image)
ref = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
ref = cv.threshold(ref, 10, 255, cv.THRESH_BINARY_INV)[1]

# find contours in the OCR-A image (i.e,. the outlines of the digits)
# sort them from left to right, and initialize a dictionary to map
# digit name to the ROI
refCnts = cv.findContours(ref.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
refCnts = refCnts[0]
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# loop over the OCR-A reference contours
for (i, c) in enumerate(refCnts):
	# compute the bounding box for the digit, extract it, and resize
	# it to a fixed size
	(x, y, w, h) = cv.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv.resize(roi, (57, 88))

	# update the digits dictionary, mapping the digit name to the ROI
	digits[i] = roi

# initialize a rectangular (wider than it is tall) and square
# structuring kernel
rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))
sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

# load the input image, resize it, and convert it to grayscale
for entry in os.listdir(directory):
	filename = os.fsdecode(entry)
	#try:
	if filename.endswith((".png", ".jpg", ".jpeg")):
		image = cv.imread(filename)
		print(directory, filename)
		if image is None:
			# i don't understand why other images are returning none.
			# from the docs:
			# The function imread loads an image from the specified file and returns it. 
			# If the image cannot be read (because of missing file, improper permissions, unsupported or invalid format),
			#  the function returns an empty matrix ( Mat::data==NULL ).
			print("error with file " + filename)
			print(directory, filename)
			#print(filename)
			try:
				filename_69 = "/Users/jackmunro/Documents/Coding/credit-card-recog/credit_card_69.jpg"
				filename_screenshot = "/Users/jackmunro/Documents/Coding/credit-card-recog/Screen Shot 2019-03-27 at 9.58.06 am.png"
				# test = cv.imread(filename_69)
				# print(test)
				# print(os.path.isfile(filename_69))
				# print(os.path.isfile(filename_screenshot))
			except:
				pass
		else:
			image = imutils.resize(image, width=300)
			gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

			image_filepath = (filename)
			# apply a tophat (whitehat) morphological operator to find light
			# regions against a dark background (i.e., the credit card numbers)
			tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, rectKernel)

			# compute the Scharr gradient of the tophat image, then scale
			# the rest back into the range [0, 255]
			gradX = cv.Sobel(tophat, ddepth=cv.CV_32F, dx=1, dy=0,
				ksize=-1)
			gradX = np.absolute(gradX)
			(minVal, maxVal) = (np.min(gradX), np.max(gradX))
			gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
			gradX = gradX.astype("uint8")

			# apply a closing operation using the rectangular kernel to help
			# cloes gaps in between credit card number digits, then apply
			# Otsu's thresholding method to binarize the image
			gradX = cv.morphologyEx(gradX, cv.MORPH_CLOSE, rectKernel)
			thresh = cv.threshold(gradX, 0, 255,
				cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

			# apply a second closing operation to the binary image, again
			# to help close gaps between credit card number regions
			thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, sqKernel)

			# find contours in the thresholded image, then initialize the
			# list of digit locations
			cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
				cv.CHAIN_APPROX_SIMPLE)
			cnts = cnts[0]
			locs = []

			# loop over the contours
			for (i, c) in enumerate(cnts):
				# compute the bounding box of the contour, then use the
				# bounding box coordinates to derive the aspect ratio
				(x, y, w, h) = cv.boundingRect(c)
				ar = w / float(h)

				# since credit cards used a fixed size fonts with 4 groups
				# of 4 digits, we can prune potential contours based on the
				# aspect ratio
				if ar > 2.5 and ar < 4.0:
					# contours can further be pruned on minimum/maximum width
					# and height
					if (w > 40 and w < 55) and (h > 10 and h < 20):
						# append the bounding box region of the digits group
						# to our locations list
						locs.append((x, y, w, h))

			# sort the digit locations from left-to-right, then initialize the
			# list of classified digits
			locs = sorted(locs, key=lambda x:x[0])
			output = []

			# loop over the 4 groupings of 4 digits
			for (i, (gX, gY, gW, gH)) in enumerate(locs):
				# initialize the list of group digits
				groupOutput = []

				# extract the group ROI of 4 digits from the grayscale image,
				# then apply thresholding to segment the digits from the
				# background of the credit card
				group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
				group = cv.threshold(group, 0, 255,
					cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

				# detect the contours of each individual digit in the group,
				# then sort the digit contours from left to right
				digitCnts = cv.findContours(group.copy(), cv.RETR_EXTERNAL,
					cv.CHAIN_APPROX_SIMPLE)
				digitCnts = digitCnts[0]
				digitCnts = contours.sort_contours(digitCnts,
					method="left-to-right")[0]

				# loop over the digit contours
				for c in digitCnts:
					# compute the bounding box of the individual digit, extract
					# the digit, and resize it to have the same fixed size as
					# the reference OCR-A images
					(x, y, w, h) = cv.boundingRect(c)
					roi = group[y:y + h, x:x + w]
					roi = cv.resize(roi, (57, 88))

					# initialize a list of template matching scores	
					scores = []

					# loop over the reference digit name and digit ROI
					for (digit, digitROI) in digits.items():
						# apply correlation-based template matching, take the
						# score, and update the scores list
						result = cv.matchTemplate(roi, digitROI,
							cv.TM_CCOEFF)
						(_, score, _, _) = cv.minMaxLoc(result)
						scores.append(score)

					# the classification for the digit ROI will be the reference
					# digit name with the *largest* template matching score
					groupOutput.append(str(np.argmax(scores)))

				# draw the digit classifications around the group
				cv.rectangle(image, (gX - 5, gY - 5),
					(gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
				cv.putText(image, "".join(groupOutput), (gX, gY - 15),
					cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

				# update the output digits list
				output.extend(groupOutput)
		try:
			print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
			print("Credit Card #: {}".format("".join(output)))
			print(image_filepath + " MATCH FOUND")
		except:
			pass
	#except:
			#pass