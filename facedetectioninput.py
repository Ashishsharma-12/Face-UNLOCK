import cv2
import numpy as np
import os
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

assure_path_exists('E:/faces')

face_classifier = cv2.CascadeClassifier('C:/Users/NILESH/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')#to extract face features

def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#convert to gray
    faces = face_classifier.detectMultiScale(gray,1.3,5)#scaling values and neighbouring values

    if faces is():
        return None
    for(x,y,w,h) in faces:
        cropped_face =  img[y:y+h,x:x+w]#crop faces
        return cropped_face

cap=cv2.VideoCapture(0)#open camera
count=0
while True:
    ret, frame=cap.read()#reading
    if face_extractor(frame) is not None:
        count +=1
        face = cv2.resize(face_extractor(frame),(400,400))#resize
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        file_name_path='E:/faces/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print('Face not found')
        pass

    if cv2.waitKey(1)==13 or count==60:
        break
        

cap.release()
cv2.destroyAllWindows()
print("Collecting Sample Complete")