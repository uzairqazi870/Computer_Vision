# Moving Pipe OCR

*************************************************************************************
Problem:
*************************************************************************************

Localize the text on moving pipe and OCR it.

*************************************************************************************
Solution:
*************************************************************************************
Steps:
	
	-> Read the input frame
	-> convert into gray scale(mono crome)
	-> compute the sobel_x (detect vertical edges), threshold it, apply morphological closing operation
	-> Find the contours
	-> Verify  area and dimensions of each contour to find the ROI
	-> extract the ROI from grayscale frame and rotate it 
	-> configure the tesseract and feed the ROI to it
	-> print the output of tesseract ( text prints on pipe)

*****************************************************
Dependent packages
*****************************************************
	-opencv-python
	-pytesseract
	-numpy 
	-argparse

*****************************************************
Installation
*****************************************************
Tesseract is an open source text recognition (OCR) Engine. It can be used directly, or (for programmers) using an API to extract printed text from images. It supports a wide variety of languages.

Head over to tesseract user manual:
https://tesseract-ocr.github.io/tessdoc/Home.html
              
Direct download link for windows:
https://github.com/UB-Mannheim/tesseract/wiki

Above step is essential if you dont want to mention the source path in every code you run

Next in your virtual environment: 
pip install pytesseract
pip install numpy
pip install argparse
pip install opencv-python

*****************************************************
