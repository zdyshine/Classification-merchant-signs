# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 20:47:24 2018

@author: h439
"""

from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

from keras.applications.inception_v3 import InceptionV3, preprocess_input
import pickle
from PIL import Image
import os, os.path
import glob

test_path = './datasets/test/'
test_file = './datasets/test.txt'

def load_image(img_path, show = False):
    img = image.load_img(img_path, target_size = (299,299))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor = preprocess_input(img_tensor)
    
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()
    return img_tensor

def write_pickle():
    test_path = './datasets/test/'
    test_file = './datasets/test.txt'
    pretrianed_model_name = 'InceptionV3'
#    result_file = './datasets/%s_06_13.csv'
    model = load_model(".//weights-improvement-78-0.99.hdf5")

    sorted_ids = list(range(1,101))
    sorted_ids.sort(key = lambda x: str(x))
    
    results = {}
    
    with open(test_file, 'r') as file:
        contents = file.readlines()
        for index, content in enumerate(contents):
            content = content.replace('\n','')
            image_path = os.path.join(test_path, content)
            
        with open(image_path, 'rb') as f:
            new_image = load_image(image_path)
            out = model.predict(new_image)
            results[image_path] = out
            
    pickle.dump(results, open('./datasets/%s_pred_test_989.pickle' % pretrianed_model_name, 'wb'))

def writr_csv():
    model4Result = './datasets/Inceptionv3_pred_test_989.pickle'
    result4 = pickle.load(open(model4Result, 'rb'))
    result_file = './datasets/inceptionv3_989_result.csv'
    
    sorted_ids = list(range(1,101))
    sorted_ids.sort(key = lambda x: str(x))
    
    with open(result_file, 'w') as file:
        for key in result4.keys():
            temp = np.asarray(result4[key])  
            file.write('%s %s\n' % (key.split('/')[-1], sorted_ids[np.argmax(temp)]))
if __name__ == "__main__":
    write_pickle()
    writr_csv()