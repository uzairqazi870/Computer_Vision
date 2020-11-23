# Import the required modules
import argparse
import cv2
import pytesseract
from pytesseract import Output
import regex as r

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("--smallcard", action='store_true', help="use if it is credit card shaped")
ap.add_argument("--bigcard", action='store_true', help="use if it is a long card")
ap.add_argument("--photocopy", action='store_true', help="use for color photocopy")
ap.add_argument( "--rotate90", action='store_true', help="rotates the image by 90")
ap.add_argument( "--rotate180", action='store_true', help="rotates the image by 180")
ap.add_argument( "--rotate270", action='store_true', help="rotates the image by 270")
args = vars(ap.parse_args())

img = args['image']
rotate_90 = args['rotate90']
rotate_180 = args['rotate180']
rotate_270 = args['rotate270']
smallcard = args['smallcard']
bigcard = args['bigcard']
photocopy = args['photocopy']

def load_process(img):
	image = cv2.imread(img) 
	image = cv2.resize(image, (640,640))

	if rotate_90:
		image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
	elif rotate_180:
		image = cv2.rotate(image, cv2.ROTATE_180_CLOCKWISE)
	elif rotate_270:
		image = cv2.rotate(image, cv2.ROTATE_270_CLOCKWISE)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 90, 150, cv2.THRESH_BINARY)[1]
	return thresh, image


def ocr_img2text(image):

	if smallcard:
		d = pytesseract.image_to_data(image, output_type=Output.DICT)
		return d

	if bigcard:
		d = pytesseract.image_to_data(image, output_type=Output.DICT)
		d1 = pytesseract.image_to_data(thresh, output_type=Output.DICT)
		return d, d1

	if photocopy:
		d = pytesseract.image_to_data(thresh, output_type=Output.DICT)
		return d

# Search for Aadhar number in the detected text using regex search pattern methods
def search_text(dict=None):
	numbers = []
	code = r.compile(r'^[2-9]{1}[0-9]{3}$')
	flag = False  

	for i, text in enumerate(d['text']):
	    result = code.search(d['text'][i])    
	    if result :
	        if not flag:        
	            numbers.append((i, text))
	            flag = True
	            i = i+1
	   
	    if flag:
	        code = r.compile(r'^[0-9]{4}$')
	        result = code.search(d['text'][i])
	        text = d['text'][i]
	        
	        if result:
	            numbers.append((i, text))
	            break
	

	if bigcard:
		numbers1 = []
		code = r.compile(r'^[2-9]{1}[0-9]{3}$')
		flag1 = True  
		flag2 = False 
		flag3 = False
		counter = 0
		for i in range(len(d1['text'])):
		    try:
		        if flag3:
		            i= i+2
		        result = code.search(d1['text'][i])  

		        if result :
		            if flag1:
		                numbers1.append((i, d1['text'][i]))
		                flag2 = True
		                flag1 = False
		                counter += 1
		                #print('1')

		        if flag2:
		            code = r.compile(r'^[0-9]{4}$')
		            i = i + 1
		            result = code.search(d1['text'][i])
		            text = d1['text'][i]

		            if result:
		                numbers1.append((i, d1['text'][i]))
		                #print("2")
		                flag1 = True
		                flag2 = False
		                counter += 1
		                flag3 = True

		        if counter == 4:
		            break
		    except:
		        pass

		return numbers, numbers1

	return numbers

# Masked the images using the location of the detected text which matched with the pattern
def output(image, numbers=None):

	rectangle_list = []

	if bigcard:
		numbers, numbers1 = numbers
	else:
		numbers = numbers

	for (i,text) in numbers:
	    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
	    rectangle_list.append([x, y, w, h])
	    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	if bigcard:
		for (i,text) in numbers1:
		    (x, y, w, h) = (d1['left'][i], d1['top'][i], d1['width'][i], d1['height'][i])
		    rectangle_list.append([x, y, w, h])
		    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	for lt in rectangle_list:
	    (x, y, w, h) = lt
	    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)

	cv2.imshow('img', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print("File closed")


if __name__ == '__main__':

	thresh, image = load_process(img)

	if bigcard:
		d, d1 = ocr_img2text(image)
		numbers, numbers1 = search_text(dict=(d,d1))
		output(image, numbers=(numbers, numbers1))

	else:
		d = ocr_img2text(image)
		numbers = search_text(dict=d)
		output(image, numbers=numbers)
    