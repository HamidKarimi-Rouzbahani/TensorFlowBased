# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:17:10 2022

@author: hk01
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%matplotlib inline

# Organize data into train, valid, test dirs
os.chdir('data/dogs-vs-cats')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')

    for i in random.sample(glob.glob('cat*'), 500):
        shutil.move(i, 'train/cat')      
    for i in random.sample(glob.glob('dog*'), 500):
        shutil.move(i, 'train/dog')
    for i in random.sample(glob.glob('cat*'), 100):
        shutil.move(i, 'valid/cat')        
    for i in random.sample(glob.glob('dog*'), 100):
        shutil.move(i, 'valid/dog')
    for i in random.sample(glob.glob('cat*'), 50):
        shutil.move(i, 'test/cat')      
    for i in random.sample(glob.glob('dog*'), 50):
        shutil.move(i, 'test/dog')

os.chdir('../../')


# GPU settings

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print("Num GPUs Available: ", len(physical_devices))
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


train_path = 'data/dogs-vs-cats/train'
valid_path = 'data/dogs-vs-cats/valid'
test_path = 'data/dogs-vs-cats/test'

# preprocessing
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10, shuffle=False)
    
#plotting some samples
imgs, labels = next(train_batches)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
plotImages(imgs)
print(labels)

# trianing the CNN
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss ='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs =10, steps_per_epoch=100, validation_steps=20, verbose = 2)


# testing on new data
test_imgs,test_labels=next(test_batches)
plotImages(test_imgs)
print(test_labels)

Predictions=model.predict(x=test_batches,verbose=0)
rounded_predcitions= np.round(Predictions)
rounded_predcitions= np.argmax(rounded_predcitions,axis=-1)

# Confusion matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ConfusionMatrixDisplay.from_predictions(test_batches.classes,rounded_predcitions)


# Fine-tuning vgg16
# download and saving vgg16 model
vgg16_model = tf.keras.applications.vgg16.VGG16()
# save only model weights
# import os.path
# if os.path.isfile('models/vgg16_model.h5') is False:
#     vgg16_model.save('models/vgg16_model.h5')
    vgg16_model.load('models/vgg16_model.h5')

vgg16_model.summary()
# what type of model this is
type(vgg16_model)

# generate a substitute model to put all the layers except for the last in
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
    
    # make all model layers untrailable
for layer in model.layers:
    layer.trainable = False

# Add a new last model layer with 2 nodes
model.add(Dense(units=2, activation='softmax'))
    
model.summary()
   
# Fine-tuning/training the new model based on vgg16
model.compile(optimizer=Adam(learning_rate=0.0001), loss ='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches,
            steps_per_epoch=len(train_batches),
            validation_data=valid_batches,
            validation_steps=len(valid_batches),
            epochs=5,
            verbose=2
)
# Testing the new fine-tuned model to check inference

Predictions=model.predict(x=test_batches,verbose=0)
rounded_predcitions= np.round(Predictions)
rounded_predcitions= np.argmax(rounded_predcitions,axis=-1)

# Confusion matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ConfusionMatrixDisplay.from_predictions(test_batches.classes,rounded_predcitions)



    











