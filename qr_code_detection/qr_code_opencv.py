import cv2
import numpy as np
import imutils
from skimage.segmentation import clear_border
import time
import argparse
import tqdm


ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True, default="./", help='path to the video')
args = vars(ap.parse_args())


def box_to_rect(box):
    
    x = min(box[0][0], box[1][0], box[2][0], box[3][0])
    x1 = max(box[0][0], box[1][0], box[2][0], box[3][0]) 
    y = min(box[0][1], box[1][1], box[2][1], box[3][1]) 
    y1 = max(box[0][1], box[1][1],box[2][1],box[3][1]) 
    w = x1 - x
    h = y1 - y
    dx = 0.1*w
    dy = 0.1*h
    x = int(x - dx)
    y = int(y - dy)
    w = int(w + 2*dx)
    h = int(h + 2*dy)
            
    return x, y, w, h

def preprocess_main(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 3, 4, 2)
    

    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction using OpenCV 
    ddepth = cv2.CV_8U
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=3)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=3)
    
    # subtract the y-gradient from the x-gradient
    gradient1 = cv2.subtract(gradX, gradY)
    gradient2 = cv2.convertScaleAbs(gradient1)
    
    # blur and threshold the image
    #blurred = cv2.blur(gradient2, (7,7))
    blurred = cv2.bilateralFilter(gradient2, 3, 4, 2)
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # perform a series of erosions and dilations
    closed1 = cv2.erode(closed, None, iterations = 5)
    closed2 = cv2.dilate(closed1, None, iterations = 5)
    
    closed3 = clear_border(closed2)
    
    return closed3




if __name__ == '__main__':
    start = time.time()
    video = args['video']

    cap = cv2.VideoCapture(video)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    vid_fps =int(cap.get(cv2.CAP_PROP_FPS))
    vid_width,vid_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video + 'output.mp4', codec, vid_fps, (640,480))



    pbar = tqdm.tqdm(total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))


    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret and frame is None:
            break

        frame = cv2.resize(frame, (640, 480))
        pro_frame = preprocess_main(frame)
        
        # find the contours in the thresholded image, then sort the contours
        # by their area, keeping only the largest one
        cnts = cv2.findContours(pro_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = sorted(cnts, key=cv2.contourArea, reverse = True)
        
        for cnt in cnts:
            if 500 < cv2.contourArea(cnt) < 7000:
                x, y, w, h = cv2.boundingRect(cnt)
                if (0.7*w) <= h <= (1.3*w):
                    # compute the rotated bounding box of the largest contour
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    x, y, w, h = box_to_rect(box)
                    #roi = frame[y:y+h, x:x+w]
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 3)

        out.write(frame)
        pbar.update(1)



    end = time.time()
    print("Time :" + str(end - start)[:4])
    
    #When everything done, release the capture
    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()