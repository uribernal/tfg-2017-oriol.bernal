import numpy as np
import h5py
from helper import DatasetManager as Dm
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam, Adadelta, Adagrad, Adamax, SGD, RMSprop
from helper import TelegramBot as Bot
import os
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model
import random
import time
from sklearn.metrics import mean_squared_error
from helper.DatasetManager import compute_pcc


# Get list with the names of the movies
movies = Dm.get_movies_names()
movies_train = movies[:-2]
movies_val = [movies[-2]]
movies_test = [movies[-1]]

db_path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/training_feat.h5'
for movie in movies_val:
    with h5py.File(db_path, 'r') as hdf:
        labels = np.array(hdf.get('dev/labels/' + movie))
        features = np.array(hdf.get('dev/features/' + movie))
    labels_val = labels.reshape(labels.shape[0], 1, labels.shape[1])[:, :, :2]
    features_val = features.reshape(features.shape[0], 1, features.shape[1])
for movie in movies_test:
    with h5py.File(db_path, 'r') as hdf:
        labels = np.array(hdf.get('dev/labels/' + movie))
        features = np.array(hdf.get('dev/features/' + movie))
    labels_test = labels.reshape(labels.shape[0], 1, labels.shape[1])[:, :, :2]
    features_test = features.reshape(features.shape[0], 1, features.shape[1])
print(labels_val.shape)
print(features_val.shape)
print(labels_test.shape)
print(features_test.shape)


def get_model():
    optimizer = Adadelta()
    input_features = Input(batch_shape=(1, 1, 7168,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(0.5)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=False, name='lsmt1')(input_dropout)
    output_dropout = Dropout(0.5)(lstm)
    output = TimeDistributed(Dense(2, activation='tanh'), name='fc')(output_dropout)

    model = Model(input=input_features, output=output)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

lstm_model = get_model()
train_loss = []
validation_loss = []
for i in range(10):
    # shuffle movies
    for j, movie in enumerate(movies):
        print('Epoch {0}, movie {1}'.format(i, j))
        with h5py.File(db_path, 'r') as hdf:
            labels = np.array(hdf.get('dev/labels/' + movie))
            features = np.array(hdf.get('dev/features/' + movie))
        labels_train = labels.reshape(labels.shape[0], 1, labels.shape[1])[:, :, :2]
        features_train = features.reshape(features.shape[0], 1, features.shape[1])

        history = lstm_model.fit(features_train,
                                 labels_train,
                                 batch_size=1,
                                 validation_data=(features_val, labels_val),
                                 verbose=2,
                                 nb_epoch=1,
                                 shuffle=False)
        lstm_model.reset_states()

train_loss.extend(history.history['loss'])
validation_loss.extend(history.history['val_loss'])
minimum_val = np.min(validation_loss)

predicted = lstm_model.predict(features_test)

# calculate root mean squared error
valenceMSE = mean_squared_error(predicted[:, 0, 0], labels_test[:, 0, 0])
print('Valence MSE = {0}\n'.format(valenceMSE))
arousalMSE = mean_squared_error(predicted[:, 0, 1], labels_test[:, 0, 1])
print('Arousal MSE = {0}\n'.format(arousalMSE))
# calculate PCC
valencePCC = compute_pcc(predicted[:, 0, 0], labels_test[:, 0, 0])
print('Valence PCC = {0}\n'.format(valencePCC))
arousalPCC = compute_pcc(predicted[:, 0, 1], labels_test[:, 0, 1])
print('Arousal PCC = {0}\n'.format(arousalPCC))

Bot.send_message('Experiment 0')
Bot.send_message('Valence MSE = {0}\n'.format(valenceMSE) +
                 'Arousal MSE = {0}\n'.format(arousalMSE) +
                 'Valence PCC = {0}\n'.format(valencePCC) +
                 'Arousal PCC = {0}\n'.format(arousalPCC))

# Path for the figures
figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/mixed_features/' + \
                   '{min:05}_Mixed_Features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.png'.format(min=minimum_val, experiment_id=0,
                                                                    batch_size=1,
                                                                    drop_out=0.5)

Bot.save_plots(train_loss, validation_loss, figures_path)

checkpoint = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/' + \
                   '{min:05}_Mixed_Features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.png'.format(min=minimum_val, experiment_id=0,
                                                                    batch_size=1,
                                                                    drop_out=0.5)
Bot.send_image(figures_path)
lstm_model.save_weights(checkpoint)