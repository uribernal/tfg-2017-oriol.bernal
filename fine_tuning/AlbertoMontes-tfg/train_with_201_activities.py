"""
This script trains an LSTM to predict emotion in videos with 201 
features extracted in each clip (16frames) with the pretrained
model from Alberto's work
"""

from helper import DatasetManager as Db
from helper import ModelGenerator as Mg
import numpy as np
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


def train_model(experiment_id, epochs, dropout_probability, batch_size, lr):

    # Path for the C3D features
    c3d_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/final/' \
               'temporal_localitzation_output_myData_resized.h5'
    store_weights_root = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/model_snapshots/'
    store_weights_file = 'FINAL_lstm_emotion_classification_{experiment_id}_e{epoch:03}.hdf5'

    # Get list with the names of the movies
    movies = Db.get_movies_names()
    # Get data & labels
    data, labels = Db.get_data_and_labels(movies, c3d_path)

    print('Data shape: {}'.format(data.shape))
    print('Labels shape: {}'.format(labels.shape))

    data = data[:26368]
    labels = labels[:26368]
    print('Data shape: {}'.format(data.shape))
    print('Labels shape: {}'.format(labels.shape))

    data = data.reshape(int(data.shape[0]), 1, data.shape[1])
    labels = labels.reshape(labels.shape[0], 1, 1)

    # Get the LSTM model
    model = Mg.lstm_alberto_tfg_activities(batch_size, dropout_probability, True)

    # Split data into train and validation
    part = 72 * 256  # 70 %
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
    model_checkpoint = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/' + \
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
    save_plots(epochs, train_loss, validation_loss, experiment_id)

'''
train_model(100,100,.5,256,1e-5)
train_model(101,500,.5,256,1e-5)
train_model(102,150,.5,32,1e-3)
train_model(103,100,.5,128,1e-5)
train_model(104,100,.5,64,1e-5)
train_model(105,100,.5,32,1e-5)
train_model(106,100,.5,256,1e-4)
train_model(107,100,.5,256,1e-5)
train_model(108,100,.5,256,1e-6)
train_model(109,500,.5,64,1e-4)
train_model(110,500,.5,32,1e-3)
train_model(111,100,.5,32,1e-3)#paciencia early stoping de 20
train_model(112,200,.5,32,1e-3)
train_model(114, 200, .5, 1, 1e-4)

'''

if __name__ == "__main__":
    import time
    from helper import TelegramBot as Bot
    experiment_id = 114
    iterations = 200
    drop_out = .5
    batch_size = 1
    lr = 1e-4

    path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/'
    file = 'FINAL_lstm_emotion_classification_{experiment_id}_e{epoch:03}.png'
    image_path = path + file.format(experiment_id=experiment_id, epoch=iterations)

    Bot.send_message('Trainning: train_model({0}, {1}, {2}, {3}, {4}'')'
                     .format(experiment_id, iterations, drop_out, batch_size, lr))
    start = time.time()
    train_model(experiment_id, iterations, drop_out, batch_size, lr)
    end = time.time()
    Bot.send_image(image_path)
    Bot.send_elapsed_time(end - start)
