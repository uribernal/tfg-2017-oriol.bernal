"""
This script trains an LSTM to predict emotion in videos with 4096 
features extracted in each clip (16frames) with the pretrained 
C3D model from Alberto's work
"""

import os
import numpy as np
from helper import DatasetManager as Db
from helper import ModelGenerator as Mg
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def save_plots(iteration, train_loss, validation_loss, experiment_id):

    path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/'
    file = 'FINAL_lstm_emotion_classification_{experiment_id}_e{epoch:03}.png'

    # Show plots
    x = np.arange(len(validation_loss))
    fig = plt.figure(1)
    fig.suptitle('LOSS', fontsize=14, fontweight='bold')

    # LOSS: TRAINING vs VALIDATION
    plt.plot(x, train_loss, '--', linewidth=2, label='train')
    plt.plot(x, validation_loss, label='validation')
    plt.legend(loc='upper right')

    plt.savefig(path+file.format(experiment_id=experiment_id, epoch=iteration), dpi=fig.dpi)
    plt.close()


def train_model(experiment_id, epochs, dropout_probability, batch_size, lr, time_steps):
    # Path for the C3D features
    c3d_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/final/C3D_features_myData_resized.h5'
    store_weights_root = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/model_snapshots/'
    store_weights_file = 'FINAL_lstm_emotion_classification_{experiment_id}_e{epoch:03}.hdf5'

    # Get list with the names of the movies
    movies = Db.get_movies_names()

    # Get data & labels
    data, labels = Db.get_data_and_labels(movies, c3d_path)

    data = data[:26368]
    labels = labels[:26368]

    data = data.reshape(int(data.shape[0]/time_steps), time_steps, data.shape[1])
    labels = labels.reshape(labels.shape[0], 1, 1)

    print('Data shape: {}'.format(data.shape))
    print('Labels shape: {}'.format(labels.shape))

    # Get the LSTM model
    model = Mg.lstm_alberto_tfg_c3d(batch_size, time_steps, dropout_probability, True)

    # Split data into train and validation
    part = 72*256  # 70 %
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

    # Callbacks
    stop_patience = 20
    model_checkpoint = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/' +\
                       store_weights_file.format(experiment_id=experiment_id, epoch=epochs)

    checkpointer = ModelCheckpoint(filepath=model_checkpoint,
                                   verbose=1,
                                   save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=5,
                                  min_lr=0,
                                  verbose=1)

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=stop_patience)

    # Training
    train_loss = []
    validation_loss = []
    #for i in range(1, epochs + 1):
        #print('Epoch {0}/{1}'.format(i, epochs))

    history = model.fit(x_train,
                            y_train,
                            batch_size=batch_size,  # Number of samples per gradient update.
                            validation_data=(x_validation, y_validation),
                            verbose=2,
                            epochs=epochs,
                            callbacks=[checkpointer, reduce_lr, early_stop],
                            shuffle=False)

    train_loss.extend(history.history['loss'])
    validation_loss.extend(history.history['val_loss'])

    print('Reseting model states')
    model.reset_states()
    #if (i % 5) == 0:
        #print('Saving snapshot...')
        #save_name = store_weights_file.format(experiment_id=experiment_id, epoch=i)
        #save_path = os.path.join(store_weights_root, save_name)
        #model.save_weights(save_path)

    save_plots(epochs, train_loss, validation_loss, experiment_id)


"""
train_model(0, 100, .5, 256, 1e-5, 1)
train_model(1, 100, .5, 32, 1e-5, 1)
train_model(2, 100, .3, 256, 1e-7, 1)
train_model(3, 100, .5, 256, 1e-3, 1)
train_model(4, 1000, .5, 32, 1e-7, 1)
train_model(5, 100, .5, 1, 1e-5, 1)
train_model(6, 100, .5, 256, 1e-4, 1)
train_model(7, 100, .5, 256, 1e-5, 1)
train_model(8, 500, .5, 256, 1e-5, 1)
train_model(9, 100, .5, 256, 1e-7, 1)
train_model(10, 100, .5, 256, 1e-3, 1)
train_model(11, 100, .5, 32, 1e-5, 1)
train_model(12, 100, .5, 32, 1e-3, 1)
train_model(13, 150, .5, 32, 1e-3, 1)
train_model(14, 500, .5, 32, 1e-3, 1)
train_model(16, 100, .5, 32, 1e-3, 1)
"""

from helper import TelegramBot as bot
import time
experiment_id = 19
iterations = 200
drop_out = .5
batch_size = 32
lr = 1e-4
time_steps = 1
path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/'
file = 'FINAL_lstm_emotion_classification_{experiment_id}_e{epoch:03}.png'
image_path = path + file.format(experiment_id=experiment_id, epoch=iterations)

bot.send_message('Trainning: train_model({0}, {1}, {2}, {3}, {4}, {5}'')'
                 .format(experiment_id, iterations, drop_out, batch_size, lr, time_steps))  # model 1
start = time.time()
train_model(experiment_id, iterations, drop_out, batch_size, lr, time_steps)
end = time.time()
bot.send_image(image_path)
bot.send_elapsed_time(end - start)
