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


def train_and_evaluate_model(data_train, labels_train, data_validation, labels_validation, batch, time_step, optimizer):
    model = model_valence_arousal(batch_size, time_step, dropout, True)
    callbacks = get_callbacks(model_checkpoint, lr_patience, stop_patience)
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


######################################## GET DATA ############################################
movies = Dm.get_movies_names()
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


######################################## CTE #################################################

cte = 0
n_folds = 1  # Number of iterations for cross-validation
lr_patience = 10  # When to decrease lr
stop_patience = 80  # When to finish trainning if no learning
timesteps = 1
dropout = 0.5

optimizers = [Adam(), SGD(), Adadelta(), RMSprop(), Adamax(), Adagrad()]
starting_lrs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
starting_lr = 0.05
optimizer = SGD(lr=starting_lr)
##############################################################################################

#################################### EXPERIMENTS #############################################
mse_valence_experiments = np.array([])
mse_arousal_experiments = np.array([])
pcc_valence_experiments = np.array([])
pcc_arousal_experiments = np.array([])

batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]
for i, (batch_size) in enumerate(batch_sizes):
    experiment_id = i + cte
    Bot.send_message('Starting experiment {0}'.format(experiment_id))

    description = 'Experiment {0}: Audio_Features, Using callbacks, drop-out={1}, batch-size={2}.'.format(
        experiment_id, dropout, batch_size)

    # Path for the weights
    store_weights_file = 'Mixed_features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.hdf5'

    model_checkpoint = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/' + \
                       store_weights_file.format(experiment_id=experiment_id, batch_size=batch_size,
                                                 drop_out=dropout)

    mse_valence = np.array([])
    mse_arousal = np.array([])
    pcc_valence = np.array([])
    pcc_arousal = np.array([])
    for i in range(n_folds):
        start = time.time()
        print("Running Fold", i + 1, "/", n_folds)
        seed = np.arange(0, labels.shape[0], dtype=np.int32)
        random.shuffle(seed)
        l = labels[seed]
        f = features[seed]

        end_index_of_training = features.shape[0] * 7 // 10
        end_index_of_validation = features.shape[0] * 9 // 10
        len_test = features.shape[0] * 1 // 10

        x_train = f[:end_index_of_training]
        y_train = l[:end_index_of_training, :, :2]
        x_validation = f[end_index_of_training:end_index_of_validation]
        y_validation = l[end_index_of_training:end_index_of_validation, :, :2]
        x_test = f[end_index_of_validation:]
        y_test = l[end_index_of_validation:, :, :2]

        model, min_val = train_and_evaluate_model(x_train, y_train, x_validation, y_validation, batch_size, timesteps, optimizer)

        predicted = model.predict(x_test)

        # calculate root mean squared error
        valenceMSE = mean_squared_error(predicted[:, 0, 0], y_test[:, 0, 0])
        print('Valence Score: %.2f MSE' % (valenceMSE))
        arousalMSE = mean_squared_error(predicted[:, 0, 1], y_test[:, 0, 1])
        print('Arousal Score: %.2f MSE' % (arousalMSE))

        # calculate PCC
        valencePCC = compute_pcc(predicted[:, 0, 0], y_test[:, 0, 0])
        print('Valence Score: %.2f MSE' % (valencePCC))
        arousalPCC = compute_pcc(predicted[:, 0, 1], y_test[:, 0, 1])
        print('Arousal Score: %.2f MSE' % (arousalPCC))

        mse_valence = np.append(mse_valence, valenceMSE)
        mse_arousal = np.append(mse_arousal, arousalMSE)
        mse_valence = np.append(mse_valence, valencePCC)
        mse_arousal = np.append(mse_arousal, arousalPCC)

        Bot.send_message(description)
        figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/mixed_features/' + \
                       '{min:05}_Mixed_Features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.png'
        image_path = figures_path.format(min=min_val, experiment_id=experiment_id, batch_size=batch_size,
                                         drop_out=dropout)
        end = time.time()
        Bot.send_image(image_path)
        Bot.send_message('Valence MSE = {0}\n' +
                         'Arousal MSE = {1}\n' +
                         'Valence PCC = {2}\n' +
                         'Arousal PCC = {3}\n'.format(mse_valence, mse_arousal, mse_valence, mse_arousal))
        Bot.send_elapsed_time(end - start)
        
        model = None  # Clearing the NN.

    mse_valence_experiments = np.append(mse_valence_experiments, np.sum(mse_valence))
    mse_arousal_experiments = np.append(mse_arousal_experiments, np.sum(mse_arousal))
    pcc_valence_experiments = np.append(pcc_valence_experiments, np.sum(pcc_valence))
    pcc_arousal_experiments = np.append(pcc_arousal_experiments, np.sum(pcc_arousal))

    Bot.send_message('FINAL VALUES FOR BATCH = {0}'.format(batch_size))
    Bot.send_message('Valence MSE = {0}\n' +
                     'Arousal MSE = {1}\n' +
                     'Valence PCC = {2}\n' +
                     'Arousal PCC = {3}\n'.format(np.sum(mse_valence), np.sum(mse_arousal), np.sum(pcc_valence), np.sum(pcc_arousal)))

    Bot.save_experiment(experiment_id, batch_size, dropout, timesteps,  starting_lr, str(optimizer.__class__)[25:-2], 'Mixed_features', lr_patience, stop_patience)
Bot.send_message('Finished')
##############################################################################################
