# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 20:07:58 2022

@author: hk01
"""

import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# Generating sample data

train_labels =[]
train_samples =[]

for i in range(50):
    # the 5% of younger who had side effects
    random_younger=randint(13,64);
    train_samples.append(random_younger)
    train_labels.append(1)
    
    
    # the 5% of older who had side effects
    random_older=randint(65,100);
    train_samples.append(random_older)
    train_labels.append(0)
    

for i in range(1000):
    # the 95% of younger who had side effects
    random_younger=randint(13,64);
    train_samples.append(random_younger)
    train_labels.append(0)
    
    
    # the 95% of older who had side effects
    random_older=randint(65,100);
    train_samples.append(random_older)
    train_labels.append(1)
    
    
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels,train_samples=shuffle(train_labels,train_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_sample = scaler.fit_transform(train_samples.reshape(-1,1))

# Constructing a feed-forward model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

# some gpu settings can be made here


# model
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')    
    ])

model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x=scaled_train_sample,y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle =True, verbose =2)

# Generate a test set for inference

test_labels =[]
test_samples =[]

for i in range(10):
    # the 5% of younger who had side effects
    random_younger=randint(13,64);
    test_samples.append(random_younger)
    test_labels.append(1)
    
    
    # the 5% of older who had side effects
    random_older=randint(65,100);
    test_samples.append(random_older)
    test_labels.append(0)
    
for i in range(200):
    # the 95% of younger who had side effects
    random_younger=randint(13,64);
    test_samples.append(random_younger)
    test_labels.append(0)
    
    
    # the 95% of older who had side effects
    random_older=randint(65,100);
    test_samples.append(random_older)
    test_labels.append(1)
    
    
test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels,test_samples=shuffle(test_labels,test_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_test_sample = scaler.fit_transform(test_samples.reshape(-1,1))

# Predict
predictions = model.predict(x=scaled_test_sample,batch_size=10, verbose =0)

rounded_predcitions= np.argmax(predictions,axis=-1)

# Confusion matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ConfusionMatrixDisplay.from_predictions(test_labels,rounded_predcitions)

# saving and loading a model
# first check to see if the model is already saved
import os.path
if os.path.isfile('models/medical_trial_model.h5') is False:
    model.save('models/medical_trial_model.h5')
    # saves archetecture, weights, training config, state of optimizer

from tensorflow.keras.models import load_model
new_model=load_model('models/medical_trial_model.h5')

new_model.summary()

new_model.get_weights()

new_model.optimizer

# less detailed saving: only architecture

# save as json
json_string = model.to_json()

# save as YAML
#yaml_string = model.to_yaml()

json_string


# model reconstruction from json
from tensorflow.keras.models import model_from_json
model_architecture=model_from_json(json_string)

model_architecture.summary()

# model reconstruction from yaml
#from tensorflow.keras.models import model_from_yaml
#model_architecture=model_from_yaml(yaml_string)

# save only model weights
# import os.path
# if os.path.isfile('models/medical_trial_model_weights.h5') is False:
#     model.save('models/medical_trial_model_weights.h5')
    model.load('models/medical_trial_model_weights.h5')



# generating a 2nd model
model2 = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')    
    ])

model2.load_weights('models/medical_trial_model_weights.h5')
model2.get_weights()

                    
#model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#model.fit(x=scaled_train_sample,y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle =True, verbose =2)







    
    
    
    















    