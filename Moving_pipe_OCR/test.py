import numpy as np
import cv2
import pytesseract
import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True, default="./Task_3_", help='path to the video')
args = vars(ap.parse_args())

class pipe_ocr:
    def __init__(self, frame):
        
        self.frame_orig = frame
        self.init_w = frame.shape[1]
        self.init_h = frame.shape[0]
        self.init_x = (self.init_w//5)
        self.init_y = (2*self.init_h)//5
        
        self.frame = frame[self.init_y : self.init_h, 
                           self.init_x : (4*self.init_w)//5]   

    def process(self):
        
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.bilateralFilter(gray, 3, 4, 2)
        sobelx = cv2.Sobel(self.gray, cv2.CV_8U, 1, 0, ksize=3)
        thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        element = cv2.getStructuringElement(cv2.MORPH_RECT, (70,13))
        morph = thresh.copy()
        cv2.morphologyEx(src=morph, op=cv2.MORPH_CLOSE, kernel=element, dst=morph)
        
        return morph
    
    def detect(self, post_process):
        contours =  cv2.findContours(post_process, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours
    
    def verify(self, contours):
        for c in contours:
            if 3500 < cv2.contourArea(c) < 4000: 
                x, y, w, h = cv2.boundingRect(c)           
                self.x = x - 10
                self.w = w + 20
                self.y = y - 5
                self.h = h + 10
                #print(self.w, self.h)
                if (245 < self.w < 255) and (30 < self.h < 35):
                    cropped = self.gray[self.y:self.y + self.h, self.x:self.x + self.w]
                    cropped = cv2.rotate(cropped, cv2.ROTATE_180)
                    cv2.rectangle(self.frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    return cropped

def build_tesseract_options(psm=7):
        # tell Tesseract to only OCR alphanumeric characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
      
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        # set the PSM mode
        options += " --psm {}".format(psm)
        # return the built options string
        return options

def ocr(roi, options):

    #pipe_details = None
    text = pytesseract.image_to_string(roi, lang="eng", config=options)
    text = text.strip()
    if len(text)==13:
        pipe_details = "{} {}.{} {} {}".format(text[0], text[1], text[2], text[3:12], text[12])
        if pipe_details is not None:
            print(pipe_details)
        return pipe_details


if __name__ == '__main__':
    start = time.time()
    video = args['video']
    options = build_tesseract_options(psm=7)
    pipe_details=None

    # cap2 = cv2.VideoCapture(video)
    # frame_width = int(cap2.get(3))
    # frame_height = int(cap2.get(4))
    # fps = cap2.get(cv2.CAP_PROP_FPS)


    # #Define the codec and create VideoWriter object 
    # fourcc = cv2.VideoWriter_fourcc(*"XVID") 
    # out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width,frame_height))

    # while cap2.isOpened():
    #     # Capture frame-by-frame
    #     ret, frame = cap2.read()
        
    #     if not ret and frame is None:
    #         break

    #     # output the frame 
    #     out.write(frame)

    # # When everything done, release the capture
    # cap2.release()
    # out.release()
    # cv2.destroyAllWindows()

    cap = cv2.VideoCapture("Task_4_Moving_pipe_OCR.mp4")

    rois = []
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret and frame is None:
            break

        pipe = pipe_ocr(frame)
        post_process = pipe.process()
        cnts = pipe.detect(post_process)
        roi = pipe.verify(cnts)
        
        if roi is not None:
            rois.append(roi)
        #     pipe_details = ocr(roi, options)
        
        # if pipe_details is not None:
        #     cv2.putText(frame, pipe_details, (100, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 2, cv2.LINE_4)
        
        end = time.time()
        # Display the resulting frame

        cv2.putText(frame, "Time :" + str(end - start)[:4], (20,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 2, cv2.LINE_4)
         
        cv2.imshow('frame', frame)
        if cv2.waitKey(17) & 0xFF == ord('q'):
            break

    #When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    for roi in rois:
        pipe_d = ocr(roi, options)
        # if pipe_d is not None:
        #     print(pipe_d)

    end2 = time.time()
    print("Time :" + str(end2 - start)[:4])


