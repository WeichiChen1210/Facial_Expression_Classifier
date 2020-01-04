# # Dataset Handler
# For the dataset "fer2013", convert the data to images, classify and output to certain label folders

from os import makedirs
from os.path import join, exists
import numpy as np
import csv
from PIL import Image


# Paths
path = './dataset/fer2013/'
csv_file = './dataset/fer2013/fer2013.csv'
train_csv = join(path, 'train.csv')
val_csv = join(path, 'val.csv')
test_csv = join(path, 'test.csv')

# Read and split data
with open(csv_file) as f:
    file = csv.reader(f)
    header = next(file)
    rows = [row for row in file]
    
    # iterate the list and check if the 'Usage' is 'Training', add label and data to list
    train_set = [row[:-1] for row in rows if row[-1] == 'Training' or row[-1] == 'PublicTest']
    csv.writer(open(train_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + train_set)
    
#     val_set = [row[:-1] for row in rows if row[-1] == 'PublicTest']
#     csv.writer(open(val_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + val_set)
    
    test_set = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
    csv.writer(open(test_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + test_set)

print(len(train_set), len(test_set))

# Convert to images
"""
folder structure:
-fer2013/
    -train/
        -0/
        -1/
        -2/
        -3/
        -4/
        -5/
        -6/
    -val/
        -0/
        -1/
        -2/
        -3/
        -4/
        -5/
        -6/
    -test/
        -0/
        -1/
        -2/
        -3/
        -4/
        -5/
        -6/
"""

train_set = join(path, 'train')
val_set = join(path, 'val')
test_set = join(path, 'test')

# for save_path, csv_file in [(train_set, train_csv), (val_set, val_csv), (test_set, test_csv)]:
for save_path, csv_file in [(train_set, train_csv), (test_set, test_csv)]:
    # create subfolder of the three sets
    if not exists(save_path):
        makedirs(save_path)
    
    num = 1
    with open(csv_file) as f:
        csv_file = csv.reader(f)
        header = next(csv_file)
        
        # convert pixels->np array->PIL image
        for i, (label, pixel) in enumerate(csv_file):
            # convert to array and reshape to 48x48
            pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
            
            # create folders of labels
            subfolder = join(save_path, label)
            if not exists(subfolder):
                makedirs(subfolder)
                
            # convert to PIL images, grayscale
            image = Image.fromarray(pixel).convert('L')
            
            # set image names with i number
            image_name = join(subfolder, '{:05d}.jpg'.format(i))
            # print(image_name)
            
            # save images
            image.save(image_name)