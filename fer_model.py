import numpy as np
import time
from os.path import join, exists
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchviz import make_dot, make_dot_from_trace

from torchsummary import summary

from PIL import Image

import matplotlib as mpl
mpl.use('tkAgg')
import matplotlib.pyplot as plt

model_type = sys.argv[1]

# Training class
class Trainer:
    def __init__(self, criterion, optimizer, device):
        self.criterion = criterion
        self.optimizer = optimizer        
        self.device = device
        self.train_accuracy = []
        self.val_accuracy = []
        self.train_sum = 0
        self.test_acc = 0
        
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
        count = 0

        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device), y.to(self.device)
            # N = X.shape[0]

            self.optimizer.zero_grad()

            # bs, ncrops, c, h, w = X.size()
            # result = model(X.view(-1, c, h, w))
            # result_avg = result.view(bs, ncrops, -1).mean(1)
            # loss = self.criterion(result_avg, y)
            outs = model(X)
            loss = self.criterion(outs, y)
            
            if step >= 0 and (step % PRINT_FREQ == 0):
                # self._state_logging(outs, y, loss, step, epoch, "Training")
                self._state_logging(outs, y, loss, step, epoch, "Training")
                count += 1
            
            loss.backward()
            self.optimizer.step()
        self.train_accuracy.append(self.train_sum / count)
        self.train_sum = 0
        
        scheduler.step()
            
    def _validate(self, model, loader, epoch, state="Validate"):
        model.eval()
        outs_list = []
        loss_list = []
        y_list = []
        
        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                # N = X.shape[0]
                
                outs = model(X)
                loss = self.criterion(outs, y)
                
                y_list.append(y)
                outs_list.append(outs)
                loss_list.append(loss)
            
            y = torch.cat(y_list)
            outs = torch.cat(outs_list)
            loss = torch.mean(torch.stack(loss_list), dim=0)
            self._state_logging(outs, y, loss, step, epoch, state)
        if state == "Validate":
            self.val_accuracy.append(self.test_acc)
                
                
    def _state_logging(self, outs, y, loss, step, epoch, state):
        acc = self._accuracy(outs, y)
        if state == "Training":
            self.train_sum += acc
        else:
            self.test_acc = acc
        print("[{:3d}/{}] {} Step {:03d} Loss {:.3f} Acc {:.3f}".format(epoch+1, EPOCHS, state, step, loss, acc))
            
    def _accuracy(self, output, target):
        batch_size = target.size(0)

        pred = output.argmax(1)
        correct = pred.eq(target)
        acc = correct.float().sum(0) / batch_size

        return acc

if __name__ == "__main__":
    # Hyperparameters
    print("Using {}...".format(model_type))

    cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }

    fer_labels = {
        'Angry': 0,
        'Disgust': 1,
        'Fear': 2,
        'Happy': 3,
        'Sad': 4,
        'Surprise': 5,
        'Neutral': 6
    }

    EPOCHS = 50
    BATCH_SIZE = 32
    PRINT_FREQ = 100
    TRAIN_NUMS = 29145
    WEIGHT_DECAY = 1e-4
    LEARNING_RATE = 1e-3

    CUDA = True

    PATH_TO_SAVE_DATA = "./"

    TRAIN_PATH = "./dataset/fer2013/train/"
    VAL_PATH = "./dataset/fer2013/val/"
    TEST_PATH = "./dataset/fer2013/test/"

    DATASET_PATH = "./dataset/fer2013/"
    MODEL_PATH = "./model/{}_model.pth".format(model_type)

    # data transforms, combinations of data augmentations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(),
            # transforms.FiveCrop((44, 44)),
            # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),            
            transforms.RandomRotation((-45, 45)),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ])
    }

    # prepare datasets
    image_datasets = {x: datasets.ImageFolder(join(DATASET_PATH, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    n_train = len(image_datasets['train'])
    n_val = len(image_datasets['val'])
    n_test = len(image_datasets['test'])
    print("training: " + str(n_train) + " images")
    print("validation: " + str(n_val) + " images")
    print("testing: " + str(n_test) + " images")

    # load datasets to dataloder
    dataloader = {}
    dataloader['train'] = DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(range(n_train)), num_workers=4)
    dataloader['val'] = DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, num_workers=4)
    dataloader['test'] = DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, num_workers=4)

    # CUDA
    # testing CUDA is available or not
    if CUDA:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # pretrained VGG model
    pretrained_vgg = None
    if model_type == "VGG19":
        pretrained_vgg = models.vgg19_bn(pretrained=True)
    elif model_type == "VGG16":
        pretrained_vgg = models.vgg16_bn(pretrained=True)
    
    # change input channel, AvgPool and classifier structure
    pretrained_vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    pretrained_vgg.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    pretrained_vgg.classifier = nn.Linear(512 * 1 * 1, 7)

    pretrained_vgg.cuda()
    # summary(pretrained_vgg, (1, 44, 44))
    
    # define loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=pretrained_vgg.parameters(),lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY) # weight_decay can be smaller
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # start training
    trainer = Trainer(criterion, optimizer, device)
    start_time = time.time()
    trainer.train_loop(pretrained_vgg, dataloader['train'], dataloader['val'])
    trainer.test(pretrained_vgg, dataloader['test'])
    end_time = time.time()

    print("--- %s sec ---" % (end_time - start_time))

    print(max(trainer.train_accuracy))
    print(max(trainer.val_accuracy))
    print(trainer.test_acc)

    # show accuracy picture
    plt.plot(range(EPOCHS), trainer.train_accuracy, 'r-', trainer.val_accuracy, 'g-')
    plt.savefig('accuracy.png')
    plt.show()
    # save model
    torch.save(pretrained_vgg.state_dict(), MODEL_PATH)
    