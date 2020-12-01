import cv2 
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, default="./Task_3_Pipe_Counting.jpg", help='path to the image')
args = vars(ap.parse_args())

def preprocess(image):
    # Load the image 
    img = cv2.imread(image) 
    #original = img.copy()
    
    #img = cv2.resize(img, (500,500), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale. 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Blur using 3 * 3 Gaussian kernel. 
    #blurred = cv2.GaussianBlur(gray, (3,3), 0)
    # to reduce noise while still maintaining the edges
    blurred = cv2.bilateralFilter(gray, 3, 4, 2)

    # Apply Hough transform on the blurred image. 
    # detected_pipes = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
    #                                     param2 = 30, minRadius = 10, maxRadius = 25)
    # detected_pipes = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 15, param1 = 30, 
    #                                     param2 = 31, minRadius = 5, maxRadius = 23)
    detected_pipes = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 15, param1 = 30, 
                                        param2 = 28, minRadius = 10, maxRadius = 23)

    return img, detected_pipes 


def draw_detected_circles(img, detected_pipes):
    radius = []

    # draw circles that are detected. 
    if detected_pipes is not None: 
        # convert the circle parameters to integers. 
        detected_pipes = np.uint16(np.around(detected_pipes))

        for pt in detected_pipes[0, :]: 
            x, y, r = pt[0], pt[1], pt[2]
            radius.append(r)

            # Draw the circumference of the circle. 
            cv2.circle(img, (x, y), r, (0, 255, 0), 2) 

            # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(img, (x, y), 1, (125, 0, 255), 3)

    rad = np.unique(radius)[np.argmax(np.unique(radius, return_counts=True)[1])]
    dia = rad * 2

    h, w = img.shape[:2]
    title = np.ones((40, w, 3), np.uint8) * 255

    cv2.putText(title, "Count : " + str(len(radius)) + "   Diameter : " + str(int(dia)), (20,32), 
    				cv2.FONT_HERSHEY_COMPLEX, 0.75 , (255,0,0), 2, cv2.LINE_4)

    result = np.vstack([title,img])

    return radius, result 
        

if __name__ == '__main__':
    img, detected_pipes = preprocess(args['image'])
    radius, img = draw_detected_circles(img, detected_pipes)
    print("Number of Pipes :", len(radius))
    cv2.imshow("Detected Pipes", img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    print("exiting...")

