"""
This script trains an LSTM to predict emotion in videos with 4096 
features extracted in each clip (16frames) with the pretrained 
C3D model from Alberto's work
"""

import h5py
import os
from fine_tuning.albertoswork import video_extraxting_fps2 as db
import numpy as np
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt


def vector_transform(vector1, l2):

    l1 = vector1.shape[0]
    l3 = vector1.shape[1]
    v = np.zeros((l2, l3))
    if l1 > l2:
        num_units = int(l1 / l2)
        res = int(l1 % l2)
        num_jumps = num_units
        vector2 = vector1[0:l1-res, :]
        for i in range(l2):
            for k in range(l3):
                x = vector2[0:num_jumps, k]
                v[i, :] = np.sum(x) / num_units
    elif(l1==l2):
        v = vector1
    else:
        raise Exception('vector1 mus be larger!')
    return v

def lstm_emotion(batch_size, time_steps, dropout_probability, summary=False):
    #model = Sequential()
    #model.add(LSTM(512, input_shape=(1, 4096)))

    '''
    A partir de l'experiment 5! (inclos)
    input_features = Input(batch_shape=(batch_size, time_steps, 4096), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(p=dropout_probability)(input_normalized)
    lstm = LSTM(2048, return_sequences=True, stateful=True, name='lsmt1')(input_dropout)
    output_dropout = Dropout(p=dropout_probability)(lstm)
    lstm2 = LSTM(1024, return_sequences=True, stateful=True, name='lsmt2')(output_dropout)
    output_dropout2 = Dropout(p=dropout_probability)(lstm2)
    lstm3 = LSTM(512, return_sequences=True, stateful=True, name='lsmt3')(output_dropout2)
    output_dropout3 = Dropout(p=dropout_probability)(lstm3)
    output = TimeDistributed(Dense(1, activation='tanh'), name='fc')(output_dropout3)
    model = Model(input=input_features, output=output)
    '''
    input_features = Input(batch_shape=(batch_size, time_steps, 4096,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(p=dropout_probability)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=True, name='lsmt1')(input_dropout)
    output_dropout = Dropout(p=dropout_probability)(lstm)
    output = TimeDistributed(Dense(1, activation='tanh'), name='fc')(output_dropout)

    model = Model(input=input_features, output=output)
    if summary:
        model.summary()
    return model

def save_data_and_labels(movies, C3D_path):
    data = np.array([])
    labels = np.array([])
    for cont, movie in enumerate(movies):
        print('Processing film: {} --------------> {}'.format(movie, cont))
        valence, arousal = db.get_labels(movie)
        labels = np.append(labels, valence, axis=0)
        with h5py.File(C3D_path, 'r') as hdf:
            # Read DB
            a = hdf.get('features')

            # Get array for the movie
            film = a.get(movie)
            print('Film shape: {}'.format(film.shape))
            print('Labels shape: {}'.format(valence.shape))

            # Reshape to adapt
            v = vector_transform(film, valence.shape[0])
            print('New film shape: {}'.format(v.shape))

            # Concatenate movies
            data = np.append(data, v)

    print('--------------------------')
    cte = int(data.shape[0]/4096)
    data = data.reshape(cte, 4096)
    # Create the HDF5 file
    hdf = h5py.File(
        '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/final/C3D_features_myData_resized.h5', 'w')
    # Create the structure of the DB
    hdf.close()
    with h5py.File(
            '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/final/C3D_features_myData_resized.h5',
            'r+') as hdf:
        # Store data
        film = hdf.create_dataset('features', data=data, compression='gzip', compression_opts=9)
    print('DATA Stored:')

def get_data_and_labels(movies, C3D_path):
    with h5py.File(C3D_path, 'r') as hdf:
        # Read DB
        data = np.array(hdf.get('features'))
    labels = np.array([])
    for movie in movies:
        valence, arousal = db.get_labels(movie)
        labels = np.append(labels, valence, axis=0)

    return  data, labels

def save_plots(iteration, train_loss, validation_loss, id):

    path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/'
    file = 'FINAL_lstm_emotion_classification_{experiment_id}_e{epoch:03}.png'

    # Show plots
    x = np.arange(iteration)
    fig = plt.figure(1)
    fig.suptitle('LOSS', fontsize=14, fontweight='bold')

    # LOSS: TRAINING vs VALIDATION
    plt.plot(x, train_loss, '--', linewidth=2, label='train')
    plt.plot(x, validation_loss, label='validation')
    plt.legend(loc='upper right')

    plt.savefig(path+file.format(experiment_id=id, epoch=iteration), dpi=fig.dpi)
    plt.close()

def train_model(experiment_id, epochs, dropout_probability, batch_size, lr, time_steps):
    # Path for the C3D features
    C3D_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/final/C3D_features_myData_resized.h5'
    store_weights_root = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/model_snapshot/'
    store_weights_file = 'FINAL_lstm_emotion_classification_{experiment_id}_e{epoch:03}.hdf5'

    # Get list with the names of the movies
    movies = db.get_movies_names()

    # Get data & labels
    data, labels = get_data_and_labels(movies, C3D_path)

    data = data[:26368]
    labels = labels[:26368]

    data = data.reshape(int(data.shape[0]/time_steps), time_steps, data.shape[1])
    labels = labels.reshape(labels.shape[0], 1, 1)

    print('Data shape: {}'.format(data.shape))
    print('Labels shape: {}'.format(labels.shape))

    # Get the LSTM model
    model = lstm_emotion(batch_size, time_steps, dropout_probability, True)


    # Split data into train and validation
    part = 72*256 # 70 %
    x_train = data[0:part, :]
    y_train = labels[0:part]
    x_validation = data[part:, :]
    y_validation = labels[part:]
    print('Train Input shape: {}'.format(x_train.shape))
    print('Train Output shape: {}\n'.format(y_train.shape))
    print('Validation Input shape: {}'.format(x_validation.shape))
    print('Validation Output shape: {}'.format(y_validation.shape))


    # Compiling Model
    print('Compiling model')
    optimizer = Adam(lr=lr)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer)
    print('Model Compiled!')

    # Training
    train_loss = []
    validation_loss = []
    #train_accuracy = []
    #validation_accuracy = []
    for i in range(1, epochs + 1):
        print('Epoch {0}/{1}'.format(i, epochs))

        history = model.fit(x_train,
                  y_train,
                  batch_size=batch_size,  # Number of samples per gradient update.
                  validation_data=(x_validation, y_validation),
                  verbose=2,  # 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
                  nb_epoch=1,
                  shuffle=False)

        train_loss.extend(history.history['loss'])
        validation_loss.extend(history.history['val_loss'])
        #train_accuracy.extend(history.history['acc'])
        #validation_accuracy.extend(history.history['val_acc'])

        print('Reseting model states')
        model.reset_states()
        if (i % 5) == 0:
            print('Saving snapshot...')
            save_name = store_weights_file.format(experiment_id=experiment_id, epoch=i)
            save_path = os.path.join(store_weights_root, save_name)
            model.save_weights(save_path)


    save_plots(epochs, train_loss, validation_loss, experiment_id)

from helper import bot
import time

'''
bot.sendMessage('Trainning: train_model(0, 100, .5, 256, 1e-5, 1)')
start = time.time()
train_model(0, 100, .5, 256, 1e-5, 1)
bot.sendImage(0, 100)
end = time.time()
elapsed = end - start
bot.sendElapsedTime(elapsed)

bot.sendMessage('Trainning: train_model(1, 100, .5, 32, 1e-5, 1)')
start = time.time()
train_model(1, 100, .5, 32, 1e-5, 1)
bot.sendImage(1, 100)
end = time.time()
elapsed = end - start
bot.sendElapsedTime(elapsed)

bot.sendMessage('Trainning: train_model(2, 100, .3, 256, 1e-7, 1)')
start = time.time()
train_model(2, 100, .3, 256, 1e-7, 1)
bot.sendImage(2, 100)
end = time.time()
elapsed = end - start
bot.sendElapsedTime(elapsed)


bot.sendMessage('Trainning: train_model(3, 100, .5, 256, 1e-5, 1)')
start = time.time()
train_model(3, 100, .5, 256, 1e-3, 1)
bot.sendImage(3, 100)
end = time.time()
elapsed = end - start
bot.sendElapsedTime(elapsed)



bot.sendMessage('Trainning: train_model(4, 100, .5, 256, 1e-5, 1)')
start = time.time()
train_model(4, 1000, .5, 32, 1e-7, 1)
bot.sendImage(4, 1000)
end = time.time()
elapsed = end - start
bot.sendElapsedTime(elapsed)

'''
bot.sendMessage('Trainning: train_model(5, 100, .5, 1, 1e-5, 1)')
start = time.time()
train_model(5, 100, .5, 1, 1e-5, 1)
bot.sendImage(5, 100)
end = time.time()
elapsed = end - start
bot.sendElapsedTime(elapsed)

bot.sendMessage('Trainning: train_model(6, 100, .5, 256, 1e-4, 1)')
start = time.time()
train_model(6, 100, .5, 256, 1e-4, 1)
bot.sendImage(6, 100)
end = time.time()
elapsed = end - start
bot.sendElapsedTime(elapsed)

'''










bot.sendMessage('Trainning: train_model(5, 100, .5, 256, 1e-5, 1)')
start = time.time()
train_model(5, 100, .5, 256, 1e-5, 1)
bot.sendImage(5, 100)
end = time.time()
elapsed = end - start
bot.sendElapsedTime(elapsed)

bot.sendMessage('Trainning: train_model(6, 500, .5, 256, 1e-5, 1)')
start = time.time()
train_model(6, 500, .5, 256, 1e-5, 1)
bot.sendImage(6, 500)
end = time.time()
elapsed = end - start
bot.sendElapsedTime(elapsed)

bot.sendMessage('Trainning: train_model(7, 100, .5, 256, 1e-7, 1)')
start = time.time()
train_model(7, 100, .5, 256, 1e-7, 1)
bot.sendImage(7, 100)
end = time.time()
elapsed = end - start
bot.sendElapsedTime(elapsed)

bot.sendMessage('Trainning: train_model(8, 500, .5, 256, 1e-7, 1)')
start = time.time()
train_model(8, 500, .5, 256, 1e-7, 1)
bot.sendImage(8, 500)
end = time.time()
elapsed = end - start
bot.sendElapsedTime(elapsed)

bot.sendMessage('Trainning: train_model(9, 100, .5, 32, 1e-7, 1)')
start = time.time()
train_model(9, 100, .5, 32, 1e-7, 1)
bot.sendImage(9, 100)
end = time.time()
elapsed = end - start
bot.sendElapsedTime(elapsed)

bot.sendMessage('Trainning: train_model(10, 500, .5, 32, 1e-7, 1)')
start = time.time()
train_model(10, 500, .5, 32, 1e-7, 1)
bot.sendImage(10, 500)
end = time.time()
elapsed = end - start
bot.sendElapsedTime(elapsed)
'''