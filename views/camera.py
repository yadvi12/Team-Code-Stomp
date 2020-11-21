import cv2
import numpy as np
import os
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
        
    def train(self):
        path = os.path.join('samples','yadvi')
        dir_images = []
        for root, dirs, files in os.walk(path):
            for i, file in enumerate(files):
                full_file_path = os.path.join(root, file)
                img = cv2.imread(full_file_path, cv2.IMREAD_GRAYSCALE)
                dir_images.append((i+1, np.asarray(img, dtype=np.uint8)))
        
        model=cv2.face_LBPHFaceRecognizer.create()
        Training_Data = []
        Labels = []
        for data in dir_images:
            label, img_array = data
            Training_Data.append(img_array)
            Labels.append(label)
        ll = np.asarray(Labels, dtype=np.int32)
        
        model.train(np.asarray(Training_Data), np.asarray(ll))
        return model

    def get_frame(self,model):                
        def face_detector(img, size=0.5):
            
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            if faces == ():
                return img, []
            
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
                roi = img[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200, 200))
            return img, roi

        ret, frame = self.video.read()
        
        image, face = face_detector(frame)
        if ret==True:
            image = cv2.flip(image,1)
        
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            results = model.predict(face)
            if results[1] < 500:
                confidence = int( 100 * (1 - (results[1])/400) )
                display_string = str(confidence) + '% Confident it is User'
                if confidence >= 83:
                    cv2.putText(image, "Verified", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                    _, jpeg = cv2.imencode('.jpg', image)
                    self.video.release()
                    return jpeg.tobytes(), 23
                
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
            
            if confidence > 845:
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                _, jpeg = cv2.imencode('.jpg', image)
                return jpeg.tobytes(), 0
                           
            else:
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                _, jpeg = cv2.imencode('.jpg', image)
                return jpeg.tobytes(), 0
    
        except:
            cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            _, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes(), 0