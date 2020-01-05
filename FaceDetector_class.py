import cv2
import os
import matplotlib.pyplot as plt
from imutils import face_utils

class FaceDetector:

	def __init__(self):
		# face_haar_path = 'haarcascade_frontalface_alt.xml'
		face_haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
		self.face_classifier = cv2.CascadeClassifier(face_haar_path)

	def FaceDetect(self,img):
		img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		faces = self.face_classifier.detectMultiScale(
			img_gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(100,100),
			flags=cv2.CASCADE_SCALE_IMAGE
		)
		return faces

