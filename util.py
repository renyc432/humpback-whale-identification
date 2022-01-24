import os
import numpy as np

import cv2
from tensorflow.keras.preprocessing import image

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def get_all_img_path(directory):
    img_path = []
    for dirname,_,filenames in os.walk(directory):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            img_path.append(fullpath)
    return img_path


def load_img_RGB(img_dir, img_filenames, width, height):
    '''
    Read an image into python and resize to the ideal shape
    
    img_dir : directory of the image
    img_filenames : name of the image
    width : width of image
    height : height of image

    Returns
    -------
    image array

    '''
    img_dataset = []
    for name in img_filenames:
        path = os.path.join(img_dir,name)
        img = image.load_img(path,target_size=(width, height))
        img = image.img_to_array(img)

        img_dataset.append(img)
    return np.array(img_dataset)


def load_img_BGR(img_dir, img_filenames, width, height):
    img_dataset = []
    for name in img_filenames:
        path = os.path.join(img_dir,name)
        image = cv2.imread(path)
        image = cv2.resize(image,(width, height))
        image = np.array(image).astype('float32')
        img_dataset.append(image)
    return np.array(img_dataset)


def split(X, y, test_size):
    '''
    Split X,y to train and validation
    
    Returns
    -------
    (X_train,y_train),(X_val,y_val)

    '''
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    for train_ind, val_ind in split.split(X,y):
        X_val,y_val = X[val_ind], y[val_ind]
        X_train,y_train = X[train_ind], y[train_ind]
    
    return X_train,y_train,X_val,y_val


# one hot encoding
# https://www.kaggle.com/pestipeti/keras-cnn-starter
def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder
