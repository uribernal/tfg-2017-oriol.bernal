import numpy as np
from helper import DatasetManager as Db
from helper import ModelGenerator as Mg
from helper import AudioHelper as Ah
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def save_plots(iteration, train_loss, validation_loss, experiment_id):

    path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/'

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

    # Path for the weights
    store_weights_file = 'Only_Audio_e_{experiment_id}_e{epoch:03}.hdf5'

    # Get list with the names of the movies
    movies = Db.get_movies_names()

    # Get data & labels
    data, labels = Db.get_data_and_labels(movies, c3d_path)

    audios = Db.get_fft_audios()
    print('Audios shape: {}'.format(audios.shape))
    print('Data shape: {}'.format(data.shape))
    print('Labels shape: {}'.format(labels.shape))

    # Get the LSTM model
    model = Mg.lstm_audio(batch_size, dropout_probability, True)


    # Split data into train and validation
    part = int(0.72*audios.shape[0])
    x_train = audios[0:part, :]
    y_train = labels[0:part]
    x_validation = audios[part:, :]
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
                                   save_best_only=True,
                                   period=5)

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

if __name__ == "__main__":

    from helper import TelegramBot as Bot
    import time
    experiment_id = 200
    iterations = 500
    drop_out = .5
    batch_size = 512
    lr = 1e-3
    time_steps = 1
    path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/audio/'
    file = 'audio_classification_{experiment_id}_e{epoch:03}.png'
    image_path = path + file.format(experiment_id=experiment_id, epoch=iterations)

    Bot.send_message('Trainning: train_model({0}, {1}, {2}, {3}, {4}, {5}'')'
                     .format(experiment_id, iterations, drop_out, batch_size, lr, time_steps))  # model 1
    start = time.time()
    train_model(experiment_id, iterations, drop_out, batch_size, lr, time_steps)
    end = time.time()
    Bot.send_image(image_path)
    Bot.send_elapsed_time(end - start)

