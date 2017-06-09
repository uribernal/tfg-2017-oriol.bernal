import numpy as np
import h5py
import os
from helper import DatasetManager as Dm
from helper import ModelGenerator as Mg
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from helper import TelegramBot as Bot
from sklearn.metrics import mean_squared_error
from helper.DatasetManager import compute_pcc


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


def train_and_evaluate_model(x_train, y_train, x_validation, y_validation, batch_size, timesteps, drop_out):
    # Path for the weights
    store_weights_file = 'Video_features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.hdf5'
    model_checkpoint = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/' + \
                       store_weights_file.format(experiment_id=experiment_id, batch_size=batch_size,
                                                 drop_out=drop_out)
    lr_patience = 10  # When to decrease lr
    stop_patience = 50  # When to finish training if no learning
    optimizer = Adam(lr=0.001)

    # Get the LSTM model
    lstm_model = Mg.lstm_alberto_tfg_c3d(batch_size, timesteps, drop_out, True)
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    print('Model Compiled!')

    train_loss = []
    validation_loss = []
    for j in range(100):
        divided_data = (x_train.shape[0] // 20) * 20

        for i in range(divided_data-1):
            history = lstm_model.fit(x_train[20*i:(20*i+20)],
                                     y_train[20*i:(20*i+20)],
                                batch_size=batch_size,
                                validation_data=(x_validation, y_validation),
                                sample_weight=None,  ######################################
                                verbose=2,
                                nb_epoch=1,
                                shuffle=False)
            print('Reseting model states')
            lstm_model.reset_states()

    train_loss.extend(history.history['loss'])
    validation_loss.extend(history.history['val_loss'])
    minimum_val = np.min(validation_loss)
    # Path for the figures
    figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/mixed_features/' + \
                   '{min:05}_Mixed_Features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.png'

    Bot.save_plots(train_loss, validation_loss, figures_path.format(min=minimum_val, experiment_id=experiment_id,
                                                                    batch_size=batch_size,
                                                                    drop_out=0.5))
    return lstm_model, minimum_val


def start_experiment(experiment_id, epochs, dropout_probability, batch_size, time_steps):

    # Get list with the names of the movies
    movies = Dm.get_movies_names()

    # Get data & labels
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
        f = np.append(f, features[:, :, :4096])

    labels = l.reshape(l.shape[0] // 3, 1, 3)
    features = f.reshape(f.shape[0] // 4096, 1, 4096)
    end_index_of_training = features.shape[0] * 70 // 100
    end_index_of_validation = features.shape[0] * 90 // 100

    x_train = features[:end_index_of_training]
    y_train = labels[:end_index_of_training, :, :1]
    x_validation = features[end_index_of_training:end_index_of_validation]
    y_validation = labels[end_index_of_training:end_index_of_validation, :, :1]
    x_test = features[end_index_of_validation:]
    y_test = labels[end_index_of_validation:, :, :1]

    print('Train Input shape: {}'.format(x_train.shape))
    print('Train Output shape: {}\n'.format(y_train.shape))
    print('Validation Input shape: {}'.format(x_validation.shape))
    print('Validation Output shape: {}\n'.format(y_validation.shape))
    print('Test Input shape: {}'.format(x_test.shape))
    print('Test Output shape: {}\n'.format(y_test.shape))

    model, min_val = train_and_evaluate_model(x_train, y_train, x_validation, y_validation, batch_size, time_steps, drop_out)

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



if __name__ == "__main__":

    from helper import TelegramBot as Bot
    import time
    experiment_id = 54
    iterations = 5000
    drop_out = .5
    batch_size = 1
    lr = 1e-3
    time_steps = 1
    description = 'Experiment {0}: Using callbacks, drop-out={1}, batch-size={2}. starting-lr={3}, model=only-visual'.format(experiment_id, drop_out, batch_size, lr)
    path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/'
    file = 'FINAL_lstm_emotion_classification_{experiment_id}_e{epoch:03}.png'
    image_path = path + file.format(experiment_id=experiment_id, epoch=iterations)

    #Bot.send_message(description)
    start = time.time()
    start_experiment(experiment_id, iterations, drop_out, batch_size, time_steps)
    end = time.time()
    #Bot.send_image(image_path)
    #Bot.send_elapsed_time(end - start)
