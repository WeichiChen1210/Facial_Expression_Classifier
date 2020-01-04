import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchviz import make_dot, make_dot_from_trace

# from torchsummary import summary

from PIL import Image

import numpy as np
import time
from os.path import join, exists

# Training class
class Trainer:
    def __init__(self, criterion, optimizer, device):
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.device = device
        
    def train_loop(self, model, train_loader, val_loader):
        for epoch in range(EPOCHS):
            print("---------------- Epoch {} ----------------".format(epoch+1))
            self._training_step(model, train_loader, epoch)
            
            self._validate(model, val_loader, epoch)
    
    def test(self, model, test_loader):
            print("---------------- Testing ----------------")
            self._validate(model, test_loader, 0, state="Testing")
            
    def _training_step(self, model, loader, epoch):
        model.train()
        
        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device), y.to(self.device)
            N = X.shape[0]
            
            self.optimizer.zero_grad()
            outs = model(X)
            loss = self.criterion(outs, y)
            
            if step >= 0 and (step % PRINT_FREQ == 0):
                self._state_logging(outs, y, loss, step, epoch, "Training")
            
            loss.backward()
            self.optimizer.step()
        
        scheduler.step()
            
    def _validate(self, model, loader, epoch, state="Validate"):
        model.eval()
        outs_list = []
        loss_list = []
        y_list = []
        
        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                N = X.shape[0]
                
                outs = model(X)
                loss = self.criterion(outs, y)
                
                y_list.append(y)
                outs_list.append(outs)
                loss_list.append(loss)
            
            y = torch.cat(y_list)
            outs = torch.cat(outs_list)
            loss = torch.mean(torch.stack(loss_list), dim=0)
            self._state_logging(outs, y, loss, step, epoch, state)
                
                
    def _state_logging(self, outs, y, loss, step, epoch, state):
        acc = self._accuracy(outs, y)
        print("[{:3d}/{}] {} Step {:03d} Loss {:.3f} Acc {:.3f}".format(epoch+1, EPOCHS, state, step, loss, acc))
            
    def _accuracy(self, output, target):
        batch_size = target.size(0)

        pred = output.argmax(1)
        correct = pred.eq(target)
        acc = correct.float().sum(0) / batch_size

        return acc

# model definition
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


if __name__ == "__main__":
    # Hyperparameters
    cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    labels = {
        'Angry': 0,
        'Disgust': 1,
        'Fear': 2,
        'Happy': 3,
        'Sad': 4,
        'Surprise': 5,
        'Neutral': 6
    }

    # MEAN = [0.49139968, 0.48215827, 0.44653124]
    # STD = [0.2023, 0.1994, 0.2010]
    MEAN = [0.49139968]
    STD = [0.2023]

    EPOCHS = 20
    BATCH_SIZE = 32
    PRINT_FREQ = 100
    TRAIN_NUMS = 29000
    LEARNING_RATE = 1e-3

    CUDA = True

    PATH_TO_SAVE_DATA = "./"

    TRAIN_PATH = "./dataset/fer2013/train/"
    VAL_PATH = "./dataset/fer2013/val/"
    TEST_PATH = "./dataset/fer2013/test/"

    DATASET_PATH = "./dataset/fer2013/"
    MODEL_PATH = "./model/model.pth"

    # data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
    }

    # prepare datasets
    image_datasets = {x: datasets.ImageFolder(join(DATASET_PATH, x), data_transforms[x]) for x in ['train', 'test']}
    n_train = len(image_datasets['train'])
    n_test = len(image_datasets['test'])
    print("training: " + str(n_train) + " images")
    print("testing: " + str(n_test) + " images")
    dataloader = {}
    dataloader['train'] = DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(range(TRAIN_NUMS)), num_workers=4)
    dataloader['val'] = DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(range(TRAIN_NUMS, n_train)), num_workers=4)
    dataloader['test'] = DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, num_workers=4)
    # dataloader = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'test']}

    # CUDA
    # testing CUDA is available or not
    if CUDA:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    # construct a model
    model = VGG(model_type='VGG11')
    model.cuda()
    # summary(model, (1, 48, 48))

    # define loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(),lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4) # weight_decay can be smaller
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # start training
    trainer = Trainer(criterion, optimizer, device)
    trainer.train_loop(model, dataloader['train'], dataloader['val'])
    trainer.test(model, dataloader['test'])

    torch.save(model.state_dict(), MODEL_PATH)


"""
new_model = VGG(model_type='VGG11')
new_model.load_state_dict(torch.load(MODEL_PATH))
new_model.eval()
new_model.cuda()


import cv2
from FaceDetector_class import *

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
# for crop_image in crop_images:
#     cv2.imshow('Face Detection',crop_image)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()

# new_img = Image.open('00000.jpg').convert('LA')
# new_img = loader(new_img).unsqueeze(1) # why squeeze 1?
# print(new_img.shape)

# print(type(crop_images[1]))
img = crop_images[0]
# cv2.imshow('Face Detection',crop_image)
# cv2.waitKey(0)
img = Image.fromarray(img)
img = img.convert('LA')
img = img.resize((48, 48))
loader = transforms.Compose([
    transforms.ToTensor()])
img = loader(img).unsqueeze(1)
print(img.shape)
img = img.to(device)
print(type(img))
out = new_model(img)

print(out)
"""
