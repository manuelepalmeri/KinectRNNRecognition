from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score)


def conv_model(datasetTrain, labelsTrain, datasetTest, labelsTest, N_ROW):
    # effettuo il reshape
    datasetTrain = datasetTrain.reshape(datasetTrain.shape[0], N_ROW, 75, 1)
    datasetTest = datasetTest.reshape(datasetTest.shape[0], N_ROW, 75, 1)

    # creo il modello
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(N_ROW, 75, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(datasetTrain, labelsTrain, batch_size=32, nb_epoch=20,
              verbose=1, validation_data=(datasetTest, labelsTest))
    score = model.evaluate(datasetTest, labelsTest, verbose=1)
    pred = model.predict_classes(datasetTest)
    return model, score, pred


def lstm_model(datasetTrain, labelsTrain, datasetTest, labelsTest, N_ROW, M):
    # creo il modello
    model = Sequential()

    model.add(LSTM(64, input_shape=(N_ROW, 75), return_sequences=True, forget_bias_init=1))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Activation('relu'))
    model.add(Dense(M))


    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(datasetTrain, labelsTrain, batch_size=64, nb_epoch=12)
    score = model.evaluate(datasetTest, labelsTest, verbose=1)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    predicted = model.predict_classes(datasetTest)

    accuracy = accuracy_score(labelsTest, predicted)
    recall = recall_score(labelsTest, predicted)
    precision = precision_score(labelsTest, predicted)
    f1 = f1_score(labelsTest, predicted)

    print('Accuracy: {}'.format(accuracy))
    print('Recall: {}'.format(recall))
    print('Precision: {}'.format(precision))
    print('F1: {}'.format(f1))
    return model, score, predicted
