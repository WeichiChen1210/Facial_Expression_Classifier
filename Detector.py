import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
from FaceDetector_class import *

# model definition
class VGG(nn.Module):
    def __init__(self, num_classes=7, model_type='VGG11'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[model_type])
        self.feature_map = []
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
	
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 7)
        )
    
    def forward(self, x):
        out = None
    
        out = self.features(x)
        # out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)
        

if __name__ == '__main__':
	model_type = 'VGG16' # must match the model type you trained

	cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

	fer_labels = [
        'Angry',
        'Disgust',
        'Fear',
        'Happy',
        'Sad',
        'Surprise',
        'Neutral'
	]
	
	# ckplus_labels = ['Neutral', 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

	# transform for input image
	loader = transforms.Compose([
			transforms.Grayscale(),
			transforms.Resize((48, 48)),
			transforms.ToTensor()
			])

	MODEL_PATH = './model/final_vgg16_fer_model.pth'
	CUDA = True
	if CUDA:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device("cpu")
	
	# load existed model
	model = VGG(7, model_type)
	model.load_state_dict(torch.load(MODEL_PATH))
	model.eval()
	model.cuda()

	# Initialize face detector
	face_detector = FaceDetector()

	# start capture video frames
	cap = cv2.VideoCapture(0)
	while True:
		# read frame, if none the skip
		ret, frame = cap.read()
		if frame is None:
			continue

		# flip the frame
		frame = cv2.flip(frame, 1)

		# detect faces
		faces = face_detector.FaceDetect(frame)
		crop_images = []
		face_pos = []

		if len(faces) > 0:
			# draw rectangle on frame, store face images and positions
			for (x,y,width,height) in faces:
				cv2.rectangle(frame, (x,y), (x + width,y + height), (0,255,0), 3)
				crop_images.append(frame[y:y + height, x:x+width])
				face_pos.append([x,y,x + width,y + height])

			# predict class of each face image and show it
			for i in range(len(crop_images)):
				# origin = np.copy(crop_images[i])

				# convert to PIL image
				crop_images[i] = Image.fromarray(crop_images[i])
				crop_images[i] = loader(crop_images[i]).unsqueeze(1)	# increase one dimension
				crop_images[i] = crop_images[i].to(device)

				# predict
				out = model(crop_images[i])

				# find the max probability of 7 classes
				classes = out.clone().cpu().detach().numpy()[0, :]
				Max = np.argmax(classes)

				# draw class label on frame
				cv2.rectangle(frame, (face_pos[i][0],face_pos[i][1] - 30), (face_pos[i][0] + 150,face_pos[i][1]), (0,255,0), -1)
				cv2.putText(frame,fer_labels[Max],(face_pos[i][0],face_pos[i][1] - 10),cv2.FONT_HERSHEY_PLAIN,2, (255,255,255), 2, cv2.LINE_AA)
				print(classes)
				print(fer_labels[Max])
		# show frame
		cv2.imshow('Demo',frame)
		# press q to exit
		if cv2.waitKey(20) & 0xff == ord('q'):
			break

	cv2.destroyAllWindows()


