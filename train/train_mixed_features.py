import numpy as np
import h5py
from helper import DatasetManager as Dm
from helper import ModelGenerator as Mg
import matplotlib.pyplot as plt
from helper import DatasetManager as Dm
from helper import ModelGenerator as Mg
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from helper import TelegramBot as Bot
import os
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model
import random
import time
import math
from sklearn.metrics import mean_squared_error


def model_valence_arousal(batch_size=1, time_step=1, dropout_probability=0.5, summary=False):
    input_features = Input(batch_shape=(batch_size, time_step, 7168,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=False, name='lsmt1')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    output = TimeDistributed(Dense(2, activation='tanh'), name='fc')(output_dropout)

    model = Model(input=input_features, output=output)

    if summary:
        model.summary()
    return model


def get_callbacks(model_checkpoint, patience1, patience2):
    checkpointer = ModelCheckpoint(filepath=model_checkpoint,
                                   verbose=1,
                                   save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=patience1,
                                  min_lr=0,
                                  verbose=1)

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience2)

    return [checkpointer, reduce_lr, early_stop]


def train_and_evaluate_model(model, data_train, labels_train, data_validation, labels_validation, batch):
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    print('Model Compiled!')

    train_loss = []
    validation_loss = []
    history = model.fit(data_train,
                        labels_train,
                        batch_size=batch,
                        validation_data=(data_validation, labels_validation),
                        sample_weight=None,  ######################################
                        verbose=2,
                        nb_epoch=10000,
                        shuffle=False,
                        callbacks=callbacks)
    print('Reseting model states')
    model.reset_states()

    train_loss.extend(history.history['loss'])
    validation_loss.extend(history.history['val_loss'])
    min_val = np.min(validation_loss)
    # Path for the figures
    figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/mixed_features/' + \
                   '{min:05}_Mixed_Features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.png'

    Bot.save_plots(train_loss, validation_loss, figures_path.format(min=min_val, experiment_id=experiment_id,
                                                                    batch_size=batch,
                                                                    drop_out=0.5))
    return model, min_val


movies = Dm.get_movies_names()
path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/training_feat.h5'
if not os.path.isfile(path):
    # Create the HDF5 file
    hdf = h5py.File(path, 'w')
    hdf.close()

experiment_id = 0
lr_patience = 10  # When to decrease lr
stop_patience = 80  # When to finish trainning if no learning

# GET DATA
l = np.array([])
f = np.array([])
for movie in movies:
    with h5py.File(path, 'r') as hdf:
        labels = np.array(hdf.get('dev/labels/' + movie))
        features = np.array(hdf.get('dev/features/' + movie))
    labels = labels.reshape(labels.shape[0], 1, labels.shape[1])
    features = features.reshape(features.shape[0], 1, features.shape[1])
    l = np.append(l, labels)
    f = np.append(f, features)
labels = l.reshape(l.shape[0] // 3, 1, 3)
features = f.reshape(f.shape[0] // 7168, 1, 7168)

batch_sizes = [128, 256, 512, 1024, 32, 64, 2048]
batch_sizes = [640]
for experiment_id, batch_size in enumerate(batch_sizes):
    experiment_id = experiment_id+1
    Bot.send_message('Start')
    time_step = 1
    dropout = 0.5
    optimizer = Adam(lr=0.001)

    description = 'Experiment {0}: Audio_Features, Using callbacks, drop-out={1}, batch-size={2}.'.format(
        experiment_id, dropout, batch_size)

    # Path for the weights
    store_weights_file = 'Mixed_features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.hdf5'

    model_checkpoint = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/' + \
                       store_weights_file.format(experiment_id=experiment_id, batch_size=batch_size,
                                                 drop_out=dropout)
    callbacks = get_callbacks(model_checkpoint, lr_patience, stop_patience)

    n_folds = 1

    mse_valence = np.array([])
    mse_arousal = np.array([])

    for i in range(n_folds):
        start = time.time()

        print("Running Fold", i+1, "/", n_folds)
        model = None # Clearing the NN.
        model = model_valence_arousal(batch_size, time_step, dropout, True)

        seed = np.arange(0, labels.shape[0], dtype=np.int32)

        random.shuffle(seed)
        labels = labels[seed]
        features = features[seed]

        len_train = features.shape[0] * 7 // 10
        len_validation = features.shape[0] * 9 // 10
        len_test = features.shape[0] * 1 // 10

        x_train = features[:len_train]
        y_train = labels[:len_train, :, :2]
        x_validation = features[len_train:len_validation]
        y_validation = labels[len_train:len_validation, :, :2]
        x_test = features[len_validation:]
        y_test = labels[len_validation:, :, :2]

        #print('Train Input shape: {}'.format(x_train.shape))
        #print('Train Output shape: {}\n'.format(y_train.shape))
        #print('Validation Input shape: {}'.format(x_validation.shape))
        #print('Validation Output shape: {}\n'.format(y_validation.shape))
        #print('Test Input shape: {}'.format(x_test.shape))
        #print('Test Output shape: {}\n'.format(y_test.shape))


        model, min_val = train_and_evaluate_model(model, x_train, y_train, x_validation, y_validation, batch_size)

        predicted = model.predict(x_test)
        print(y_test.shape)
        print(predicted.shape)

        # calculate root mean squared error
        valenceScore = mean_squared_error(predicted[:, 0, 0], y_test[:, 0, 0])
        print('Valence Score: %.2f RMSE' % (valenceScore))
        testScore = mean_squared_error(predicted[:, 0, 1], y_test[:, 0, 1])
        print('Arousal Score: %.2f RMSE' % (testScore))


        print('VALENCE Test prediction shape: {0}, sum = {1}'.format(predicted.shape, valenceScore))

        print('AROUSAL Test prediction shape: {0}, sum = {1}'.format(predicted.shape, testScore))

        mse_valence = np.append(mse_valence, valenceScore)
        mse_arousal = np.append(mse_arousal, testScore)

        Bot.send_message(description)
        figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/mixed_features/' + \
                       '{min:05}_Mixed_Features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.png'
        image_path = figures_path.format(min=min_val, experiment_id=experiment_id, batch_size=batch_size, drop_out=dropout)
        end = time.time()
        Bot.send_image(image_path)
        Bot.send_elapsed_time(end - start)



    print(mse_valence)
    print(mse_arousal)
    Bot.send_message('MSE valence: {}'.format(mse_valence))
    Bot.send_message('MSE arousal: {}'.format(mse_arousal))

    Bot.send_message('Finished')

    for i in range(predicted.shape[0]):
        print('{}   {}'.format(predicted[i,0,0], y_test[i,0,0]))
        print('   {}'.format(predicted[i,0,0], y_test[i,0,0]))