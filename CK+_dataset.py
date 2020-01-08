# # Dataset Handler
# For the dataset "CK+", obtain image labels and copy images to certain folders

from os import makedirs, listdir
from os.path import join, exists
import shutil
import math

if __name__ == "__main__":
    ROOT = './dataset/CK+/'
    LABEL = ROOT + 'Emotion_labels'
    IMAGES = ROOT + 'CK+images'
    train_folder = ROOT + 'train/'
    test_folder = ROOT + 'test/'
    all_image_folder = ROOT + 'images/'
    
    # check the folders exist or not
    for path in [train_folder, test_folder, all_image_folder]:
        if not exists(path):
                makedirs(path)
        for i in range(7):
            labels = path + str(i)
            if not exists(labels):
                makedirs(labels)
    
    people = listdir(LABEL)
    print(len(people))
    
    # in label folder, for each person(sxxx folder), find the expression sequences
    for person in people:
        person_path = join(LABEL, person)
        # path for image folder
        image_person_path = join(IMAGES, person)

        # get expression folder path
        expressions = listdir(person_path)

        # for each expression, get the label files
        for expression in expressions:
            expr_path = join(person_path, expression)
            image_expr_path = join(image_person_path, expression)

            # check if there's label file in this expression
            label_file = listdir(expr_path)
            if len(label_file) == 0:    # if not, then skip this expression sequence
                continue
            else:   # else read the label of this expression seq and copy the certain images to its belonging class folder
                file_path = join(expr_path, label_file[0])
                image_file_name = label_file[0][:-12]   # path of image file

                label = 0
                with open(file_path, 'r') as f:
                    label = int(eval(f.read().strip()))
                # ignore label 2
                if label >= 3:
                    label -= 1
                elif label == 2:
                    continue

                image_path = join(image_expr_path, image_file_name) + '.png'    # full path of image file
                
                # get all frames and sort by their names reversly
                frames = listdir(image_expr_path)
                frames.sort(reverse=True)
                
                # get the 5 frames from the peak frame and copy
                if len(frames) < 5:
                    print("< 5")
                else:
                    for i in range(5):
                        name = frames[i]
                        image_path = join(image_expr_path, name)
                        target_path = all_image_folder + str(label) + '/' + name

                        shutil.copy(image_path, target_path)

                # also copy neural frames
                cut = image_file_name[:-2] + "01" + '.png'
                image_path = join(image_expr_path, cut)
                target_path = all_image_folder + str(0) + '/' + cut
                
                shutil.copy(image_path, target_path)
    
    # so far all the images are in CK+images folder, next split to training and test sets

    # count the number of images in each classes
    count_list = []
    for i in range(7):
        path = all_image_folder + str(i) + '/'
        files = listdir(path)
        count = len(files)
        count_list.append(count)
    print(sum)

    split_num = 1666

    # split to training and test sets
    for i in range(7):
        classes = str(i) + '/'
        path = all_image_folder + classes
    
        files = listdir(path)
        count = 0
        split_num = math.floor(count_list[i] * 0.9)
        for image in files:
            src_path = join(path, image)
            count += 1
            if count <= split_num:
                train_path = train_folder + classes + image
                # print(src_path)
                # print(train_path)
                shutil.copy(src_path, train_path)
            else:
                test_path = test_folder + classes + image
                shutil.copy(src_path, test_path)
        

