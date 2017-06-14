import numpy as np
import h5py
from helper import DatasetManager as Dm
from helper import ModelGenerator as Mg
import matplotlib.pyplot as plt
from helper import DatasetManager as Dm
from helper import ModelGenerator as Mg
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam, Adadelta, Adagrad, Adamax, SGD, RMSprop
from helper import TelegramBot as Bot
import os
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model
import random
import time
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from helper.DatasetManager import compute_pcc


figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/mixed_features/' + \
               '{min:05}_Mixed_Features_e_{experiment_id}_b{batch_size:03}_d{dropout:02}_fold{n_fold}.png'

store_weights_file = 'Mixed_features_e_{experiment_id}_b{batch_size:03}_d{dropout:02}.hdf5'


def model_valence_arousal(batch_size=1, time_step=1, dropout_probability=0.5, summary=False):
    input_features = Input(batch_shape=(batch_size, time_step, 7168,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=True, name='lsmt1')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    output = TimeDistributed(Dense(2, activation='tanh'), name='fc')(output_dropout)

    model = Model(input=input_features, output=output)

    if summary:
        model.summary()
    return model


def train_and_evaluate_model(experiment_id, optimizer, batch_size, timesteps, dropout):
    mse_valence = np.array([])
    mse_arousal = np.array([])
    pcc_valence = np.array([])
    pcc_arousal = np.array([])

    # Get the model
    lstm_model = model_valence_arousal(batch_size, timesteps, dropout, True)
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    print('Model Compiled!')

    # Get data Train, Validation and Test
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


    # Start Training
    train_loss = []
    validation_loss = []
    for i in range(150):
        # shuffle movies
        for j, movie in enumerate(movies_train):
            print('Epoch {0}, movie {1}'.format(i, j))

            # Get training data
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

    fig_path = figures_path.format(min=minimum_val, experiment_id=experiment_id, batch_size=batch_size, dropout=dropout,
                                   n_fold=1)
    Bot.save_plots(train_loss, validation_loss, fig_path)

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

    scores = []
    scores.append(np.mean(valenceMSE))
    scores.append(np.mean(arousalMSE))
    scores.append(np.mean(valencePCC))
    scores.append(np.mean(arousalPCC))

    return scores, fig_path


def get_optimizer(opt, star_lr):
    optimizer = Adam(lr=star_lr)
    if opt == 'Adadelta':
        optimizer = Adadelta(lr=star_lr)
    elif opt == 'SGD':
        optimizer = SGD(lr=star_lr)
    elif opt == 'RMSprop':
        optimizer = RMSprop(lr=star_lr)
    elif opt == 'Adamax':
        optimizer = Adamax(lr=star_lr)
    elif opt == 'Adagrad':
        optimizer = Adagrad(lr=star_lr)
    return optimizer


def train(optimizer, batch_size, timesteps, dropout, starting_lr=1e-3, lr_patience=10, stop_patience=50):
    optimizer = get_optimizer(optimizer, starting_lr)

    start, experiment_id = Bot.start_experiment()

    scores, fig_path = train_and_evaluate_model(experiment_id, optimizer, batch_size, timesteps, dropout)

    Bot.save_experiment(optimizer, batch_size, timesteps, dropout, 0, starting_lr, lr_patience, stop_patience,
                        'Mixed Features', 1, 512, scores)

    Bot.end_experiment(start, fig_path, scores)


if __name__ == '__main__':
    train('Adam', 1, 1, 0.5, starting_lr=1e-4, lr_patience=0, stop_patience=0)
