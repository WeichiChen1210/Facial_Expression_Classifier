import cv2
from FaceDetector_class import *

if __name__ == '__main__':
	face_detector = FaceDetector()
	img = cv2.imread('test2.jpg')
	faces = face_detector.FaceDetect(img)
	print(len(faces))

	crop_images = []
	if len(faces) > 0:
		for (x,y,width,height) in faces:
			#cv2.rectangle(img, (x,y), (x + width,y + height), (255,0,0), 3)
			crop_images.append(img[y:y + height, x:x+width])
	print(len(crop_images))
	for crop_image in crop_images:
		cv2.imshow('Face Detection',crop_image)
		cv2.waitKey(0)
	cv2.destroyAllWindows()


