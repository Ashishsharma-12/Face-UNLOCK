import cv2
import numpy as np
from os import listdir#os module class used for fetching data from directory
from os.path import isfile, join

model = cv2.face.LBPHFaceRecognizer_create()
model.read('E:/trained.yml')
face_classifier = cv2.CascadeClassifier('C:/Users/NILESH/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')#to extract face features

def face_detector(img,size = 0.5):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray,1.3,5)
	if faces is():
		return img,[]

	for(x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)#image,actual values, changed values, color of rect.,thickness
		roi = img[y:y+h,x:x+w]
		roi = cv2.resize(roi,(200,200))

	return img,roi

cap = cv2.VideoCapture(0)
while True:
	ret,frame = cap.read()

	image,face = face_detector(frame)

	try:
		face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
		result = model.predict(face)

		if  result[1] < 500:
			confidence = int(100*(1-(result[1])/300))
			display_string = str(confidence)+"% confidence it is user"
		
		cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(255,120,255),2)
		
		if confidence > 81:
			cv2.putText(image,"Unlocked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
			cv2.imshow('Face Cropper',image)
		
		else:
			cv2.putText(image,"locked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
			cv2.imshow('Face Cropper',image)


	except:
		cv2.putText(image,"Face not found",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
		cv2.imshow('Face Cropper',image)
		pass

	if cv2.waitKey(1)==13:
		break
	
cap.release()
cv2.destroyAllWindows()