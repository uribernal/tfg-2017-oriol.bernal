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


def train_and_evaluate_model(experiment_id, optimizer, batch_size, timesteps, dropout, n_folds, features, labels, lr_patience, stop_patience, figures_path):

    mse_valence = np.array([])
    mse_arousal = np.array([])
    pcc_valence = np.array([])
    pcc_arousal = np.array([])
    for i in range(n_folds):
        print("Running Fold", i + 1, "/", n_folds)
        seed = np.arange(0, labels.shape[0], dtype=np.int32)
        random.shuffle(seed)
        l = labels[seed]
        f = features[seed]

        end_index_of_training = features.shape[0] * 7 // 10
        end_index_of_validation = features.shape[0] * 9 // 10

        x_train = f[:end_index_of_training]
        y_train = l[:end_index_of_training, :, :2]
        x_validation = f[end_index_of_training:end_index_of_validation]
        y_validation = l[end_index_of_training:end_index_of_validation, :, :2]
        x_test = f[end_index_of_validation:]
        y_test = l[end_index_of_validation:, :, :2]

        model = model_valence_arousal(batch_size, timesteps, dropout, True)
        # Path for the weights
        store_weights_file = 'Mixed_features_e_{experiment_id}_b{batch_size:03}_d{dropout:02}.hdf5'

        model_checkpoint = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/' + \
                           store_weights_file.format(experiment_id=experiment_id, batch_size=batch_size,
                                                     dropout=dropout)

        callbacks = get_callbacks(model_checkpoint, lr_patience, stop_patience)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        print('Model Compiled!')

        train_loss = []
        validation_loss = []
        history = model.fit(x_train, y_train,
                            batch_size = batch_size,
                            validation_data = (x_validation, y_validation),
                            verbose = 2,
                            nb_epoch = 10000,
                            shuffle = False,
                            callbacks = callbacks)
        print('Reseting model states')
        #model.reset_states()

        train_loss.extend(history.history['loss'])
        validation_loss.extend(history.history['val_loss'])
        minimum_val = np.min(validation_loss)

        figures_path = figures_path.format(min=minimum_val, experiment_id=experiment_id, batch_size=batch_size, dropout=dropout, n_fold=i)
        Bot.save_plots(train_loss, validation_loss, figures_path)

        predicted = model.predict(x_test)

        # calculate root mean squared error
        valenceMSE = mean_squared_error(predicted[:, 0, 0], y_test[:, 0, 0])
        print('Valence MSE = {0}\n'.format(valenceMSE))
        arousalMSE = mean_squared_error(predicted[:, 0, 1], y_test[:, 0, 1])
        print('Arousal MSE = {0}\n'.format(arousalMSE))

        # calculate PCC
        valencePCC = compute_pcc(predicted[:, 0, 0], y_test[:, 0, 0])
        print('Valence PCC = {0}\n'.format(valencePCC))
        arousalPCC = compute_pcc(predicted[:, 0, 1], y_test[:, 0, 1])
        print('Arousal PCC = {0}\n'.format(arousalPCC))

        Bot.send_message('Valence MSE = {0}\n'.format(valenceMSE) +
                         'Arousal MSE = {0}\n'.format(arousalMSE) +
                         'Valence PCC = {0}\n'.format(valencePCC) +
                         'Arousal PCC = {0}\n'.format(arousalPCC))

        mse_valence = np.append(mse_valence, valenceMSE)
        mse_arousal = np.append(mse_arousal, arousalMSE)
        pcc_valence = np.append(pcc_valence, valencePCC)
        pcc_arousal = np.append(pcc_arousal, arousalPCC)

    scores = []
    scores.append(np.mean(mse_valence))
    scores.append(np.mean(mse_arousal))
    scores.append(np.mean(pcc_valence))
    scores.append(np.mean(pcc_arousal))

    return scores, figures_path


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


def train(optimizer, batch_size, timesteps, dropout, n_folds=1, starting_lr=1e-3, lr_patience=10, stop_patience=50):
    optimizer = get_optimizer(optimizer, starting_lr)

    ######################################## GET DATA ############################################
    movies = Dm.get_movies_names()
    figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/mixed_features/' + \
                   '{min:05}_Mixed_Features_e_{experiment_id}_b{batch_size:03}_d{dropout:02}_fold{n_fold}.png'
    path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/training_feat.h5'
    if not os.path.isfile(path):
        # Create the HDF5 file
        hdf = h5py.File(path, 'w')
        hdf.close()

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
    ##############################################################################################


    #################################### EXPERIMENTS #############################################
    start, experiment_id = Bot.start_experiment()

    scores, figures_path = train_and_evaluate_model(experiment_id, optimizer, batch_size, timesteps, dropout, n_folds, features, labels, lr_patience, stop_patience, figures_path)

    Bot.save_experiment(optimizer, batch_size, timesteps, dropout, n_folds, starting_lr, lr_patience, stop_patience, 'Mixed Features', 1, 512, scores)

    Bot.end_experiment(start, figures_path, scores)



if __name__ == '__main__':
    #train('Adam', 64, 1, 0.5, n_folds=1, starting_lr=1e-1, lr_patience=10, stop_patience=30)
    train('Adam', 64, 1, 0.5, n_folds=1, starting_lr=1e-2, lr_patience=10, stop_patience=30)
    train('Adam', 64, 1, 0.5, n_folds=1, starting_lr=1e-3, lr_patience=10, stop_patience=30)
    train('Adam', 64, 1, 0.5, n_folds=1, starting_lr=1e-4, lr_patience=10, stop_patience=30)
    train('Adam', 64, 1, 0.5, n_folds=10, starting_lr=1e-3, lr_patience=10, stop_patience=30)
