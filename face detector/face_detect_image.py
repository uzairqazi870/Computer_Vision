import cv2
import argparse
from face_detector import Face_Detector

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', default='cascade/haarcascade_frontalface_default.xml', help='path to the face cascade')
ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

fd = Face_Detector(args['face'])
faces = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

print("%d face(s) detected" %(len(faces)))

for (x,y,w,h) in faces:
	cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

cv2.imshow("Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
