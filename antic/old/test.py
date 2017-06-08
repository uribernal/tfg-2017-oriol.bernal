import numpy as np
import h5py
from helper import DatasetManager as Dm
from helper import ModelGenerator as Mg
import matplotlib.pyplot as plt
import numpy as np
from helper import DatasetManager as Dm
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

    plt.savefig(path + file.format(experiment_id=experiment_id, batch_size=batch_size), dpi=fig.dpi)
    plt.close()
    plt.close()


def train_model(experiment_id, epochs, dropout_probability, batch_size, lr):


    # Path for the weights
    store_weights_file = 'train_with_Mix_features_e_{experiment_id}_b{batch_size:03}.hdf5'

    movies = Dm.get_movies_names()

    with h5py.File('/home/uribernal/Desktop/MediaEval2017/data/data/emotional_impact.h5', 'r') as hdf:
        labels = np.array(hdf.get('labels'))
        # video_features = np.array(hdf.get('video_features'))
        audio_features = np.array(hdf.get('audio_features'))

    with h5py.File('/home/uribernal/Desktop/MediaEval2017/data/data/provisional_video_features.h5', 'r') as hdf:
        video_features = np.array(hdf.get('features'))

    audio_features = audio_features.reshape(audio_features.shape[0], 3 * 2 * 512)
    video_features = video_features.reshape(video_features.shape[0], video_features.shape[1])

    labels = labels.reshape(labels.shape[0], 1, 3)
    labels = labels[:, :, :2]

    print('Labels shape: {}'.format(labels.shape))
    print('Video features shape: {}'.format(video_features.shape))
    print('Audio features shape: {}'.format(audio_features.shape))

    data = np.append(video_features, audio_features, axis=1)
    data = data.reshape(video_features.shape[0], 1, video_features.shape[1]+audio_features.shape[1])
    print('Labels shape: {}'.format(labels.shape))
    print('Data shape: {}'.format(data.shape))

    # Get the LSTM model
    model = Mg.lstm_alberto_tfg_c3d_val_ar(batch_size, 1, dropout_probability, True)

    # Make data fit into batches
    res = data.shape[0] % batch_size
    new_len = data.shape[0] - res
    data = data[:new_len]
    labels = labels[:new_len]
    print('Audios shape: {}'.format(data.shape))
    print('Labels shape: {}\n'.format(labels.shape))

    # Make data fit into batches
    res = data.shape[0] % batch_size
    new_len = data.shape[0] - res
    audios = data[:new_len]
    labels = labels[:new_len]
    print('Audios shape: {}'.format(audios.shape))
    print('Labels shape: {}\n'.format(labels.shape))

    # Split data into train and validation
    num = int(0.7 * new_len / batch_size)
    part = num * batch_size  # 70 %
    x_train = audios[0:part, :]
    y_train = labels[0:part]
    x_validation = audios[part:, :]
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
    stop_patience = 100
    model_checkpoint = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/' +\
                       store_weights_file.format(experiment_id=experiment_id, batch_size=batch_size)

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
    print(history.history.keys())
    train_loss.extend(history.history['loss'])
    validation_loss.extend(history.history['val_loss'])

    print('Reseting model states')
    model.reset_states()
    save_plots(epochs, train_loss, validation_loss, experiment_id)


if __name__ == "__main__":
    from helper import TelegramBot as Bot
    import time
    experiment_id = 1000000
    iterations = 5000
    drop_out = .5
    batch_size = 512
    lr = 1e-3
    for experiment_id in range(1,100):
        batch_size = 32 * experiment_id
        path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/'
        file = 'train_with_Mix_features_{experiment_id}_b{batch_size:03}.png'
        image_path = path + file.format(experiment_id=experiment_id, batch_size=batch_size)

        description = 'Experiment {0}: Using callbacks, drop-out={1}, batch-size={2}. starting-lr={3}, model=mix_features'.format(
            experiment_id, drop_out, batch_size, lr)

        Bot.send_message(description)
        start = time.time()
        train_model(experiment_id, iterations, drop_out, batch_size, lr)
        end = time.time()
        Bot.send_image(image_path)
        Bot.send_elapsed_time(end - start)
    Bot.send_message('FINISHED')



