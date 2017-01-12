import pandas as pd
import numpy as np
import glob
import random
import matplotlib.pyplot as plt
from prediction_model import conv_model, lstm_model
from keras.utils import np_utils
import functions
random.seed(34898)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score, confusion_matrix)

# inserisco in una lista i nomi dei file
thankyouFiles = glob.glob("Dataset/thankyou/*.csv")
maybeFiles = glob.glob("Dataset/maybe/*.csv")
neverFiles = glob.glob("Dataset/never/*.csv")
wonderfulFiles = glob.glob("Dataset/wonderful/*.csv")
babyFiles = glob.glob("Dataset/baby/*.csv")
worldFiles = glob.glob("Dataset/world/*.csv")
rainbowFiles = glob.glob("Dataset/rainbow/*.csv")
littleFiles = glob.glob("Dataset/little/*.csv")

folders = [thankyouFiles, maybeFiles, neverFiles, wonderfulFiles, babyFiles, worldFiles, rainbowFiles, littleFiles]
N_movements = len(folders)
N_ROW = 130 #mean_value(folders)
# nomi dei joint ordinati (saranno le colonne del dataset)
'''
jointName = ['SpineBase_x', 'SpineBase_y', 'SpineBase_z', 'SpineMid_x', 'SpineMid_y', 'SpineMid_z', 'Neck_x', 'Neck_y','Neck_z',
             'Head_x', 'Head_y', 'Head_z', 'ShoulderLeft_x', 'ShoulderLeft_y', 'ShoulderLeft_z', 'ElbowLeft',
             'WristLeft', 'HandLeft', 'ShoulderRight', 'ElbowRight', 'WristRight',
             'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft',
             'HipRight', 'KneeRight', 'AnkleRight', 'FootRight', 'SpineShoulder',
             'HandTipLeft', 'ThumbLeft', 'HandTipRight', 'ThumbRight'] '''



# popolo il dataset prendendo i valori dai file
(datasetTrain, datasetTest, labelsTrain, labelsTest) = functions.import_data(folders, N_ROW)

#labelsTrain = [random.randint(0, 3) for i in len(labelsTrain)]
# effettuo i reshape necessari per darli in input alla rete neurale
Y_train = np_utils.to_categorical(labelsTrain)
Y_test = np_utils.to_categorical(labelsTest)

#model, score, pred = conv_model(datasetTrain, labelsTrain, datasetTest, labelsTest, N_ROW)
#ac = []

# model, score, pred = lstm_model(datasetTrain, labelsTrain, datasetTest, labelsTest, N_ROW, len(folders))
model = Sequential()

model.add(LSTM(64, input_shape=(N_ROW, 75), return_sequences=True, forget_bias_init="one"))
model.add(Activation('relu'))
model.add(LSTM(64, return_sequences=True))
model.add(Activation('relu'))
model.add(LSTM(64))
model.add(Dropout(0.25))

model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(N_movements))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(datasetTrain, Y_train, batch_size=64, nb_epoch=25)
score = model.evaluate(datasetTest, Y_test, verbose=1)

print 'Test score:', score[0]
print 'Test accuracy:', score[1]

predicted = model.predict_classes(datasetTest)


functions.stat(predicted, labelsTest, N_movements)

accuracy = accuracy_score(labelsTest, predicted)
recall = recall_score(labelsTest, predicted, average="weighted")
precision = precision_score(labelsTest, predicted, average="weighted")
f1 = f1_score(labelsTest, predicted, average="weighted")
confusion = confusion_matrix(labelsTest, predicted)

print('Accuracy: {}'.format(accuracy))
print('Recall: {}'.format(recall))
print('Precision: {}'.format(precision))
print('F1: {}'.format(f1))
print("Confusion:")
print(confusion)

#print 'Test score:', score[0]
#print 'Test accuracy:', score[1]
