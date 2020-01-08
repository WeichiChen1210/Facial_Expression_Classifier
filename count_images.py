import sys
from os.path import join, exists
from os import listdir
import numpy as np
import matplotlib as mpl
mpl.use('tkAgg')
import matplotlib.pyplot as plt

file_path = sys.argv[1]

if __name__ == "__main__":
    path = './dataset/' + file_path

    if 'fer2013' in file_path:
        labels = 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
    elif 'CK+' in file_path:
        labels = 'Neutral', 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'
    
    count_list = []

    # count the numbers of each class
    for i in range(7):
        dirpath = join(path, str(i))
        files = listdir(dirpath)
        count_list.append(len(files))
    
    print(count_list)
    # draw pie and percentage
    plt.pie(count_list, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')
    plt.savefig('distribution.png')
    plt.show()