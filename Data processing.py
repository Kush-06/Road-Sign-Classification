#256x256 pixels around 55000 images
#not evenly split between categories

import os
import cv2
import random 
import numpy as np
import pickle
import matplotlib.pyplot as plt 


dataset_path = os.path.join(os.getcwd(), "uk_roadsigns")
categories = os.listdir(dataset_path)
training_data = []


def get_num_datapoints ():
    for category in categories:
        path = os.path.join(dataset_path,category)
        length = len(os.listdir(path))
        print(category, ": ", length)

def process_images():
    for category in categories:
        folder_path = os.path.join(dataset_path,category)
        count = 0
        number_of_images = len(os.listdir(folder_path))
        for img in os.listdir(folder_path):
            img_array = cv2.imread(os.path.join(folder_path,img), cv2.IMREAD_GRAYSCALE)
            #resize images from 256x256 to 85x85
            resized_array = cv2.resize(img_array, (85,85))
            training_data.append([resized_array, category])
            count += 1
            print('\r', count , "/" , number_of_images, end = '', flush=True)

        print("\n", category, " complete")

def save_data():
    x = []
    y=[]

    for image, label in training_data:
        x.append(image)
        y.append(categories.index(label))

    x = np.array(x).reshape(-1,85,85,1)
    y = np.array(y)

    pickle_out= open(os.path.join(os.getcwd(), "features.pickle"), "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()

    pickle_out= open(os.path.join(os.getcwd(), "labels.pickle"), "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close() 

    print("Date saved!")

process_images()
random.shuffle(training_data)
save_data()
