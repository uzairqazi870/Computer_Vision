import cv2
import argparse
from face_detector import Face_Detector
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', default='cascade/haarcascade_frontalface_default.xml', help='path to face file')
ap.add_argument('-e', '--eyes', default='cascade/haarcascade_eye.xml', help='path to eyes file')
ap.add_argument('-i', '--video', help='path to optional video file')
args = vars(ap.parse_args())


fd = Face_Detector(args['face'], args['eyes'])

if not args['video'] :
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args['video'])

if not camera.isOpened:
		print("Failed to load the frames. Exiting ....")

while True:
	ret, frame = camera.read()

	if frame is None and not ret:
		break

	frame = imutils.resize(frame, width=500)

	#image = cv2.resize(frame, (480,480))
	
	result = fd.detect(frame)



	cv2.imshow("faces", result)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

print("exiting....")
camera.release()
cv2.destroyAllWindows()




