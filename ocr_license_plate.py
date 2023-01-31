# import the necessary packages
from sakura.anpr import anpr 
from imutils import paths
import argparse
import imutils
import cv2
import dbcheck

def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-c", "--clear-border", type=int, default=-1,
	help="whether or to clear border pixels before OCR'ing")
ap.add_argument("-p", "--psm", type=int, default=7,
	help="default PSM mode for OCR'ing license plates")
ap.add_argument("-d", "--debug", type=int, default=-1,
	help="whether or not to show additional visualizations")
args = vars(ap.parse_args())

# initialize our ANPR class
anpr = anpr.Sakura(debug=args["debug"] > 0)
# grab all image paths in the input directory
# imagePaths = sorted(list(paths.list_images(args["input"])))
# imagePaths = paths.list_images(args["input"])
imagePaths = args["input"]


# loop over all image paths in the input directory
# for imagePath in imagePaths:
	# load the input image from disk and resize it
image = cv2.imread(imagePaths)
image = imutils.resize(image, width=600, height=600)
# apply automatic license plate recognition
#image = imutils.resize(image, width=400, height=400)
image = cv2.bilateralFilter(image, 3, 105, 105)
anpr.debug_imshow("Bilateral Filter", image)

(lpText, lpCnt) = anpr.find_and_ocr(image, psm=args["psm"],
	clearBorder=args["clear_border"] > 0)
# only continue if the license plate was successfully OCR'd
if lpText is not None and lpCnt is not None:
	# fit a rotated bounding box to the license plate contour and
	# draw the bounding box on the license plate
	box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
	box = box.astype("int")
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	# compute a normal (unrotated) bounding box for the license
	# plate and then draw the OCR'd license plate text on the
	# image
	(x, y, w, h) = cv2.boundingRect(lpCnt)
	cv2.putText(image, cleanup_text(lpText), (x, y - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	# show the output ANPR image
	print("[Number Plate: ] {}".format(lpText))
	# print(lpText)
	lpText2 = lpText.strip()
	#print(lpText2)
	is_available = dbcheck.check_item_availability(lpText2)

	if is_available:
		print("plate exist")
	else:
		print("plate not exist")
	cv2.imshow("Output ANPR", image)
	cv2.waitKey(0)

