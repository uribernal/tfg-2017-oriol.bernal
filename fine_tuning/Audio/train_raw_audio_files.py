import matplotlib.pyplot as plt
import numpy as np
from helper import DatasetManager as Db
from helper import ModelGenerator as Mg
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam


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

    # MIN
    val, idx = min((val, idx) for (idx, val) in enumerate(validation_loss))
    plt.annotate(str(val), xy=(idx, val), xytext=(idx, val - 0.01),
                 arrowprops=dict(facecolor='black', shrink=0.0005))

    plt.savefig(path + file.format(experiment_id=experiment_id, epoch=iteration), dpi=fig.dpi)
    plt.close()
    plt.close()


def train_model(experiment_id, epochs, dropout_probability, batch_size, lr):
    # Path for the C3D features
    c3d_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/final/C3D_features_myData_resized.h5'

    # Path for the weights
    store_weights_file = 'Only_Audio_e_{experiment_id}_e{epoch:03}.hdf5'

    # Get list with the names of the movies
    movies = Db.get_movies_names()

    # Get data & labels
    data, labels = Db.get_data_and_labels(movies, c3d_path)
    labels = labels.reshape(labels.shape[0], 1, 1)

    audios = Db.get_fft_audios()
    print('Audios shape: {}'.format(audios.shape))
    print('Data shape: {}'.format(data.shape))
    print('Labels shape: {}'.format(labels.shape))

    # Get the LSTM model
    model = Mg.lstm_raw_audio(batch_size, True)

    # Make data fit into batches
    res = audios.shape[0] % batch_size
    new_len = audios.shape[0] - res
    audios = audios[:new_len]
    labels = labels[:new_len]
    print('Audios shape: {}'.format(audios.shape))
    print('Labels shape: {}\n'.format(labels.shape))
    #audios = audios.reshape(new_len, 1, 98*64)
    audios = audios.reshape(new_len, 1, 98 * 64)
    # Split data into train and validation
    num = int(0.7 * new_len / batch_size)
    part = num * batch_size  # 70 %
    x_train = audios[0:part, :, :]
    y_train = labels[0:part]
    x_validation = audios[part:, :, :]
    y_validation = labels[part:]
    print('Train Input shape: {}'.format(x_train.shape))
    print('Train Output shape: {}\n'.format(y_train.shape))
    print('Validation Input shape: {}'.format(x_validation.shape))
    print('Validation Output shape: {}\n'.format(y_validation.shape))

   
    # Compiling Model
    print('Compiling model')
    optimizer = Adam(lr=lr)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer)
    print('Model Compiled!')

    # Callbacks
    stop_patience = 80
    model_checkpoint = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/' +\
                       store_weights_file.format(experiment_id=experiment_id, epoch=epochs)

    checkpointer = ModelCheckpoint(filepath=model_checkpoint,
                                   verbose=1,
                                   save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=10,
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
    experiment_id = 5
    iterations = 5000
    drop_out = .5
    batch_size = 512*2
    lr = 1e-3
    path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/'
    file = 'audio_classification_{experiment_id}_e{epoch:03}.png'
    image_path = path + file.format(experiment_id=experiment_id, epoch=iterations)

    description = 'Experiment {0}: Using callbacks, drop-out={1}, batch-size={2}. starting-lr={3}, model=only-acoustic'.format(experiment_id, drop_out, batch_size, lr)

    #Bot.send_message(description)
    start = time.time()
    train_model(experiment_id, iterations, drop_out, batch_size, lr)
    end = time.time()
    #Bot.send_image(image_path)
    #Bot.send_elapsed_time(end - start)

