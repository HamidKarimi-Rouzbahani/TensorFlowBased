# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:59:54 2022

@author: hk01
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
#%matplotlib inline

# Download and save mobilenet
mobile = tf.keras.applications.mobilenet.MobileNet()
# import os.path
# if os.path.isfile('models/mobilenet_model.h5') is False:
#     mobile.save('models/mobilenet_model.h5')
# mobile=model.load('models/mobilenet_model.h5')


def prepare_image(file):
    img_path = 'data/MobileNet-samples/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


# tryin the accuracy of the mobilenet on some sample natural images first 
preprocessed_image = prepare_image('1.jpg')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
results

preprocessed_image = prepare_image('2.jpg')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
results

preprocessed_image = prepare_image('3.jpg')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
results

# Now we work with a sign language dataset to fine-tune mobile net on
# Organize data into train, valid, test dirs
os.chdir('data/Sign-Language-Digits-Dataset')
if os.path.isdir('train/0/') is False: 
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')

    for i in range(0, 10):
        shutil.move(f'{i}', 'train')
        os.mkdir(f'valid/{i}')
        os.mkdir(f'test/{i}')

        valid_samples = random.sample(os.listdir(f'train/{i}'), 30)
        for j in valid_samples:
            shutil.move(f'train/{i}/{j}', f'valid/{i}')

        test_samples = random.sample(os.listdir(f'train/{i}'), 5)
        for k in test_samples:
            shutil.move(f'train/{i}/{k}', f'test/{i}')
os.chdir('../..')



train_path = 'data/Sign-Language-Digits-Dataset/train'
valid_path = 'data/Sign-Language-Digits-Dataset/valid'
test_path = 'data/Sign-Language-Digits-Dataset/test'

# preprocessing
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)
    
mobile.summary()

# Grab the output from the sixth to last layer of the model and store it in this variable x
x = mobile.layers[-6].output

# now we generate a fully connected last layer to replace the last six which we remove from mobile
# the x is used when a model is functional rather than sequential
output = Dense(units=10, activation='softmax')(x)

# construct the final model by determining the input and output
model = Model(inputs=mobile.input, outputs=output)

# we decided based on trial and error to freeze the last 23 (out of 88) layers in training
for layer in model.layers[:-23]:
    layer.trainable = False

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss ='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches,
            steps_per_epoch=len(train_batches),
            validation_data=valid_batches,
            validation_steps=len(valid_batches),
            epochs=10,
            verbose=2
)

# testing on new data
test_labels = test_batches.classes
predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

rounded_predcitions= np.round(predictions)
rounded_predcitions= np.argmax(rounded_predcitions,axis=-1)

# Confusion matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ConfusionMatrixDisplay.from_predictions(test_labels,rounded_predcitions)



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))










