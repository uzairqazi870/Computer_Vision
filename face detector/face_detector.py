import cv2


class Face_Detector:
    def __init__(self, face_cascade_path, eyes_cascade_path):
        self.face_detect = cv2.CascadeClassifier(face_cascade_path)
        self.eyes_detect = cv2.CascadeClassifier(eyes_cascade_path)
    
    def detect(self, frame, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # create a copy of the original frame
        clone = frame.copy()
        
        faces = self.face_detect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                                 minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
        
        if len(faces)>0:
            # loop over the faces
            for face in faces:
                (x,y,w,h) = face
                face_center = (x + w//2, y + h//2)
                cv2.ellipse(clone, face_center, (w//2,h//2), 0, 0, 360, (255,0,0), 2, cv2.LINE_AA)

                # crop the face from the frame
                ROI = gray[y:y+h, x:x+w]
                # find the eyes in the face
                eyes = self.eyes_detect.detectMultiScale(ROI)
                # loop over each eye
                for eye in eyes:
                    (x1,y1,w1,h1) = eye
                    eye_center = (x+x1+w1//2, y+y1+h1//2)
                    cv2.ellipse(clone, eye_center, (w1//2, h1//2), 0, 0, 360, (0,255,0), 2, cv2.LINE_4)
        
        
        return clone