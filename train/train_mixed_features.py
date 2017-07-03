import numpy as np
import h5py
from helper import DatasetManager as Dm
from keras.optimizers import Adam, Adadelta, Adagrad, Adamax, SGD, RMSprop
from helper import TelegramBot as Bot
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model
from sklearn.metrics import mean_squared_error
from helper.DatasetManager import compute_pcc

figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/mixed_features/' + \
               'Mixed_Features_e_{experiment_id}_b{batch_size:03}_d{dropout:02}_fold{n_fold}.png'

store_weights_file = 'Mixed_features_e_{experiment_id}_b{batch_size:03}_d{dropout:02}.hdf5'

db_path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/training_feat.h5'


def model_valence_arousal(batch_size=1, time_step=1, dropout_probability=0.5, opt=None, summary=False):
    input_features = Input(batch_shape=(batch_size, time_step, 7168,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=True, name='lsmt1')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    output = TimeDistributed(Dense(2, activation='tanh'), name='fc')(output_dropout)

    model = Model(input=input_features, output=output)
    if opt is None:
        opt = Adam()

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    print('Model Compiled!')

    if summary:
        model.summary()
    return model


def get_data(data_split):

    # Get data Train, Validation and Test
    movies = Dm.get_movies_names()

    movies_val = movies[data_split+1]
    movies_test = movies[data_split]
    del movies[data_split]
    del movies[data_split]
    movies_train = movies
    print(movies_test)
    print(movies_val)
    print(movies_train)
    with h5py.File(db_path, 'r') as hdf:
        labels = np.array(hdf.get('dev/labels/' + movies_val))
        features = np.array(hdf.get('dev/features/' + movies_val))
    labels_val = labels.reshape(labels.shape[0], 1, labels.shape[1])[:, :, -1].reshape(labels.shape[0], 1, 1)
    features_val = features.reshape(features.shape[0], 1, features.shape[1])
    with h5py.File(db_path, 'r') as hdf:
        labels = np.array(hdf.get('dev/labels/' + movies_test))
        features = np.array(hdf.get('dev/features/' + movies_test))
    labels_test = labels.reshape(labels.shape[0], 1, labels.shape[1])[:, :, -1].reshape(labels.shape[0], 1, 1)
    features_test = features.reshape(features.shape[0], 1, features.shape[1])

    return movies_train, features_val, labels_val, features_test, labels_test


def train_and_evaluate_model(experiment_id, optimizer, batch_size, timesteps, dropout, data_split):

    # Get the model
    lstm_model = model_valence_arousal(batch_size, timesteps, dropout, optimizer, True)

    # Get the data
    movies_train, features_val, labels_val, features_test, labels_test = get_data(data_split)


    # Start Training
    t_loss = np.array([])
    v_loss = np.array([])
    train_loss = []
    validation_loss = []

    t_acc = np.array([])
    v_acc = np.array([])
    train_acc = []
    validation_acc = []
    for i in range(10):
        # shuffle movies
        for j, movie in enumerate(movies_train):
            print('Epoch {0}, movie {1}'.format(i, j))

            # Get training data
            with h5py.File(db_path, 'r') as hdf:
                labels = np.array(hdf.get('dev/labels/' + movie))
                features = np.array(hdf.get('dev/features/' + movie))
            labels_train = labels.reshape(labels.shape[0], 1, labels.shape[1])[:, :, -1].reshape(labels.shape[0], 1, 1)
            features_train = features.reshape(features.shape[0], 1, features.shape[1])

            history = lstm_model.fit(features_train,
                                     labels_train,
                                     batch_size=batch_size,
                                     validation_data=(features_val, labels_val),
                                     verbose=2,
                                     nb_epoch=1,
                                     shuffle=False)
            lstm_model.reset_states()

            train_loss.extend(history.history['loss'])
            validation_loss.extend(history.history['val_loss'])
            train_acc.extend(history.history['acc'])
            validation_acc.extend(history.history['val_acc'])

        t_loss = np.append(t_loss, np.mean(train_loss[-28:]))
        v_loss = np.append(v_loss, np.mean(validation_loss[-28:]))
        t_acc = np.append(t_acc, np.mean(train_acc[-28:]))
        v_acc = np.append(v_acc, np.mean(validation_acc[-28:]))

    fig_path = figures_path.format(experiment_id=experiment_id, batch_size=batch_size, dropout=dropout,
                                   n_fold=1)

    # Predict with de TEST set
    predicted = lstm_model.predict(features_test, batch_size=1)

    # calculate root mean squared error
    valenceMSE = mean_squared_error(predicted[:, 0, 0], labels_test[:, 0, 0])
    print('Valence MSE = {0}\n'.format(valenceMSE))
    arousalMSE = mean_squared_error(predicted[:, 0, 1], labels_test[:, 0, 1])
    print('Arousal MSE = {0}\n'.format(arousalMSE))

    # calculate PCC
    valencePCC = compute_pcc(predicted[:, 0, 0], labels_test[:, 0, 0])
    print('Valence PCC = {0}\n'.format(valencePCC))
    arousalPCC = compute_pcc(predicted[:, 0, 1], labels_test[:, 0, 1])
    print('Arousal PCC = {0}\n'.format(arousalPCC))

    scores = []
    scores.append(np.mean(valenceMSE))
    scores.append(np.mean(arousalMSE))
    scores.append(np.mean(valencePCC))
    scores.append(np.mean(arousalPCC))

    # Store the plot figure
    Bot.save_plots(t_loss, v_loss, fig_path)
    json_file = lstm_model.to_json()
    with open('/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/models/model_{}.json'.format(experiment_id), 'w') as f:
        f.write(json_file)
    lstm_model.save_weights('/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/models/weights_{}.json'.format(experiment_id))
    return scores, fig_path


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


def train(optimizer, batch_size, timesteps, dropout, starting_lr=1e-3, lr_patience=10, stop_patience=50, data_split=1):

    start, experiment_id = Bot.start_experiment()
    print('Experiment: {}'.format(experiment_id))
    scores, fig_path = train_and_evaluate_model(experiment_id, optimizer, batch_size, timesteps, dropout, data_split)

    # Create JSON file and XLS with experiment info
    Bot.save_experiment(optimizer, batch_size, timesteps, dropout, 0, starting_lr, lr_patience, stop_patience,
                        'Mixed Features', 1, 512, scores)

    # Send elapsed time and results
    Bot.end_experiment(start, fig_path, scores)


if __name__ == '__main__':
    train('Adam', 1, 1, 0.5, starting_lr=1e-3, lr_patience=0, stop_patience=0, data_split=0)

