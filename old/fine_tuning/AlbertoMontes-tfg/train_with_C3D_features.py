"""
This script trains an LSTM to predict emotion in videos with 4096 
features extracted in each clip (16frames) with the pretrained 
C3D model from Alberto's work
"""

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam

from old.helper import DatasetManager as Db
from old.helper import ModelGenerator as Mg


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

    # MIN
    val, idx = min((val, idx) for (idx, val) in enumerate(validation_loss))
    plt.annotate(str(val), xy=(idx, val), xytext=(idx, val-0.01),
                arrowprops=dict(facecolor='black', shrink=0.0005))

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

    res = data.shape[0] % batch_size
    new_len = data.shape[0] - res
    data = data[:new_len]
    labels = labels[:new_len]

    data = data.reshape(int(data.shape[0]/time_steps), time_steps, data.shape[1])
    labels = labels.reshape(labels.shape[0], 1, 1)

    print('Data shape: {}'.format(data.shape))
    print('Labels shape: {}'.format(labels.shape))

    # Get the LSTM model
    model = Mg.lstm_alberto_tfg_c3d(batch_size, time_steps, dropout_probability, True)
    #model = Mg.three_layers_lstm(2048, 1024, 512, batch_size, time_steps, dropout_probability, True)
    #model = Mg.two_layers_lstm(2048, 512, batch_size, time_steps, dropout_probability, True)

    # Split data into train and validation
    num = int(0.7 * new_len / batch_size)
    part = num*batch_size  # 70 %
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
    stop_patience = 100
    model_checkpoint = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/' +\
                       store_weights_file.format(experiment_id=experiment_id, epoch=epochs)

    checkpointer = ModelCheckpoint(filepath=model_checkpoint,
                                   verbose=1,
                                   save_best_only=True)
                                   #period=5)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=5,######
                                  min_lr=1e-10000,
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
train_model(20, 200, .5, 1, 1e-4, 1)
train_model(21, 200, .5, 1, 1e-2, 1)#stateful = false


"""

if __name__ == "__main__":

    import time
    experiment_id = 54
    iterations = 5000
    drop_out = .5
    batch_size = 256
    lr = 1e-3
    time_steps = 1
    description = 'Experiment {0}: Using callbacks, drop-out={1}, batch-size={2}. starting-lr={3}, model=only-visual'.format(experiment_id, drop_out, batch_size, lr)
    path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/'
    file = 'FINAL_lstm_emotion_classification_{experiment_id}_e{epoch:03}.png'
    image_path = path + file.format(experiment_id=experiment_id, epoch=iterations)

    #Bot.send_message(description)
    start = time.time()
    train_model(experiment_id, iterations, drop_out, batch_size, lr, time_steps)
    end = time.time()
    #Bot.send_image(image_path)
    #Bot.send_elapsed_time(end - start)
