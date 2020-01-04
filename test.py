import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
from FaceDetector_class import *

model_type = 'VGG16' # must match the model type you trained

cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

labels = [
        'Angry',
        'Disgust',
        'Fear',
        'Happy',
        'Sad',
        'Surprise',
        'Neutral'
	]

loader = transforms.Compose([
			transforms.Grayscale(),
			# transforms.Resize((48, 48)),
			transforms.ToTensor()
			])

class VGG(nn.Module):
    def __init__(self, num_classes=7, model_type='VGG11'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[model_type])
        self.feature_map = []
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 7),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        out = None
    
        out = self.features(x)
        out = self.avgpool(out)
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
	MODEL_PATH = './model/model.pth'
	CUDA = True
	if CUDA:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device("cpu")
	
	# load model
	model = VGG(7, model_type)
	model.load_state_dict(torch.load(MODEL_PATH))
	model.eval()
	model.cuda()

	# face detector
	start_time = time.time()
	face_detector = FaceDetector()
	img = cv2.imread('./test images/test2.jpg')
	faces = face_detector.FaceDetect(img)

	crop_images = []
	if len(faces) > 0:
		for (x,y,width,height) in faces:
			#cv2.rectangle(img, (x,y), (x + width,y + height), (255,0,0), 3)
			crop_images.append(img[y:y + height, x:x+width])
	print(len(crop_images))

	# predict class of each image and show it
	for img in crop_images:
		origin = np.copy(img)

		# convert to PIL image
		img = Image.fromarray(img)
		img = loader(img).unsqueeze(1)	# increase one dimension
		img = img.to(device)
		# predict
		out = model(img)

		# find the max probability of 7 classes
		classes = out.clone().cpu().detach().numpy()[0, :]
		max = np.argmax(classes)
		print(classes)
		print(labels[max])

		cv2.imshow('Face Detection',origin)
		cv2.waitKey(0)

	cv2.destroyAllWindows()