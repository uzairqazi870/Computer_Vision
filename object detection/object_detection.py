import cv2
import numpy as np 
import argparse
import time

ap = argparse.ArgumentParser(description="Object detection")
ap.add_argument('--video', help="path to video file (optional)")
ap.add_argument('--camera', help="Specify camera address", type=int, default=0)
args = ap.parse_args()

def color_threshold():
	img = np.zeros((300,300,3), np.uint8)
	cv2.namedWindow("image")

	def value(x):
	    return x

	cv2.createTrackbar("R","image",0,255, value)
	cv2.createTrackbar("G","image",0,255, value)
	cv2.createTrackbar("B","image",0,255, value)

	color_store = []
	count=0
	while (1):
	    cv2.imshow("image", img)
	        
	    r = cv2.getTrackbarPos("R","image")
	    g = cv2.getTrackbarPos("G","image")
	    b = cv2.getTrackbarPos("B","image")
	    
	    img[:] = [b,g,r]
	    
	    k = cv2.waitKey(1) & 0xff
	    if k == ord('s'):
	        if count==0:
	            print("lower_threshold:", (b,g,r))
	            color_store.append([b,g,r])
	            count+=1
	        elif count==1:
	            print("upper_threshold:", (b,g,r))
	            color_store.append([b,g,r])
	            cv2.destroyAllWindows()
	            return color_store
	        
	    if k == ord('q'):
	        print(b,g,r)
	        color_store.append((b,g,r))
	        cv2.destroyAllWindows()
	        break

color_store = color_threshold()
lowerBlue = np.array(color_store[0], np.uint8)
upperBlue = np.array(color_store[1], np.uint8)


if not args.video:
	cap = cv2.VideoCapture(args.camera)
	if not cap.isOpened:
		print("failed to open camera")
		exit(0)
else:
	cap = cv2.VideoCapture(args.video)


while True:
	ret, frame = cap.read()

	if frame is None:
		break

	blue = cv2.inRange(frame, lowerBlue, upperBlue)
	blue = cv2.GaussianBlur(blue, (3,3), 0)

	(_, cnts, _) = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if len(cnts)>0:
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
		rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnts)))
		cv2.drawContours(frame, [rect], -1, (0,255,0), 2)


	cv2.imshow("Tracking", frame)

	if cv2.waitKey(1) & 0xff == ord('q'):
		break

print("exiting...")
cap.release()
cv2.destroyAllWindows()



