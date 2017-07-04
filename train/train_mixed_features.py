import numpy as np
import h5py
from helper import DatasetManager as Dm
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.optimizers import Adam
from keras.models import Model
from helper import TelegramBot as Bot
from sklearn.metrics import mean_squared_error
from helper.DatasetManager import compute_pcc
from helper.ExperimentHelper import Experiment
import time


np.set_printoptions(threshold=np.NaN)
figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/mixed_features/' + \
                   'Mixed_Features_e_{experiment_id}_b{batch_size:03}_c{cells}.png'

store_weights_file = 'Mixed_features_e_{experiment_id}_b{batch_size:03}_c{cells}.hdf5'

db_path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/training_feat.h5'


def get_model(cells=512, bs=1, ts=1, i_dim=7168, dp=0.5, opt=None, lr=1e-8, summary=False):

    input_features = Input(batch_shape=(bs, ts, i_dim,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dp)(input_normalized)
    lstm = LSTM(cells, return_sequences=True, stateful=True, name='lsmt1')(input_dropout)
    middle_output = Dropout(dp)(lstm)
    middle_dense = TimeDistributed(Dense(512, activation='tanh'), name='fc1')(middle_output)
    output_dropout = Dropout(dp)(middle_dense)
    output = Dense(2, activation='tanh', name='fc')(output_dropout)

    model = Model(input=input_features, output=output)
    if opt is None:
        opt = Adam(lr=lr)

    model.compile(loss='mean_squared_error', optimizer=opt)
    print('Model Compiled!')
    '''
    input_features = Input(batch_shape=(bs, ts, i_dim,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dp)(input_normalized)
    lstm = LSTM(cells, return_sequences=True, stateful=True, name='lsmt1')(input_dropout)
    middle_dropout = Dropout(dp)(lstm)
    lstm2 = LSTM(cells//2, return_sequences=True, stateful=True, name='lsmt2')(middle_dropout)
    output_dropout = Dropout(dp)(lstm2)
    output = TimeDistributed(Dense(2, activation='tanh'), name='fc')(output_dropout)

    model = Model(input=input_features, output=output)
    if opt is None:
        opt = Adam(lr=lr)

    model.compile(loss='mean_squared_error', optimizer=opt)
    print('Model Compiled!')
    '''
    '''
        print('Build STATEFUL model...')
        model = Sequential()
        model.add(LSTM(cells, batch_input_shape=(bs, ts, i_dim), return_sequences=False, stateful=True))
        model.add(Dropout(dp))
        model.add(Dense(2, activation='tanh'))

        if opt is None:
            opt = Adam()
        model.compile(loss='mean_squared_error', optimizer=opt)
        print('Model Compiled!')

        if summary:
            model.summary()
        return model
    '''

    if summary:
        model.summary()
    return model


def get_data(split):
    # Get data Train, Validation and Test
    movies = Dm.get_movies_names()

    movies_val = [movies[split + 1], movies[split + 2], movies[split + 3]]
    movies_test = [movies[split], movies[split + 4]]
    del movies[split]
    del movies[split]
    del movies[split]
    del movies[split]
    del movies[split]
    movies_train = movies

    return movies_train, movies_val, movies_test


def train_and_evaluate_model(experiment_id, num_epochs, cells, opt, bs, ts, dp, lr, split):

    # Path for results
    fig_path = figures_path.format(experiment_id=experiment_id, batch_size=bs, cells=cells)

    # Get the model
    model = get_model(cells=cells, bs=bs, ts=ts, dp=dp, opt=opt, lr=lr, summary=True)

    # Get the data
    movies_train, movies_val, movies_test = get_data(split)

    print('Train...')
    training_loss_epochs = []
    test_loss_epochs = []
    for epoch in range(num_epochs):
        start = time.time()
        tr_loss_movies = []
        for movie in movies_train:
            with h5py.File(db_path, 'r') as hdf:
                labels = np.array(hdf.get('dev/labels/' + movie))
                features = np.array(hdf.get('dev/features/' + movie))
                y_train = labels.reshape(labels.shape[0], 1, labels.shape[1])[:, :, :2]
            x_train = features.reshape(features.shape[0], 1, features.shape[1])[:, :, :]
            for i in range(len(x_train)):
                x = np.expand_dims(np.expand_dims(x_train[i][0], axis=0), axis=0)
                y_true = np.array([y_train[i, :, :]])
                tr_loss = model.train_on_batch(x, y_true)
                tr_loss_movies.append(tr_loss)
            model.reset_states()

        val_loss_movies = []
        for movie in movies_val:
            with h5py.File(db_path, 'r') as hdf:
                labels = np.array(hdf.get('dev/labels/' + movie))
                features = np.array(hdf.get('dev/features/' + movie))
            y_val = labels.reshape(labels.shape[0], 1, labels.shape[1])[:, :, :2]
            x_val = features.reshape(features.shape[0], 1, features.shape[1])[:, :, :]
            for i in range(len(x_val)):
                x = np.expand_dims(np.expand_dims(x_val[i][0], axis=0), axis=0)
                y_true = np.array([y_val[i, :, :]])
                val_loss = model.test_on_batch(x, y_true)
                val_loss_movies.append(val_loss)
            model.reset_states()

        training_loss_epochs = np.append(training_loss_epochs, np.mean(tr_loss_movies))
        test_loss_epochs = np.append(test_loss_epochs, np.mean(val_loss_movies))

        print('epoch = {}, loss training = {}, loss testing = {}, elapsed time = {}'
              .format(epoch, np.mean(tr_loss_movies), np.mean(val_loss_movies), time.time()-start))
        print('___________________________________')
        Bot.send_message('epoch = {}, loss training = {}, loss testing = {}, elapsed time = {}'
                         .format(epoch, np.mean(tr_loss_movies), np.mean(val_loss_movies), time.time() - start))
    # Save results
    Dm.save_plots(training_loss_epochs, test_loss_epochs, fig_path)
    model.save('/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/models/model_{}.h5'.format(
                experiment_id))
    json_file = model.to_json()
    with open('/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/models/model_{}.json'.format(
                experiment_id), 'w') as f:
        f.write(json_file)

    model.save_weights('/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/models/weights_{}.h5'.format(
                experiment_id))

    valence_mse = np.array([])
    arousal_mse = np.array([])
    valence_pcc = np.array([])
    arousal_pcc = np.array([])

    # Predict with de TEST set
    for movie in movies_test:
        with h5py.File(db_path, 'r') as hdf:
            labels = np.array(hdf.get('dev/labels/' + movie))
            features = np.array(hdf.get('dev/features/' + movie))
            y_test = labels.reshape(labels.shape[0], 1, labels.shape[1])[:, :, :2]
        x_test = features.reshape(features.shape[0], 1, features.shape[1])[:, :, :]

        predicted = model.predict(x_test, batch_size=1)

        if len(predicted.shape) < 3:
            # calculate root mean squared error
            valence_mse = np.concatenate(valence_mse, mean_squared_error(predicted[:, 0], y_test[:, 0, 0]))
            print('Valence MSE = {0}\n'.format(valence_mse))
            arousal_mse = np.concatenate(arousal_mse, mean_squared_error(predicted[:, 1], y_test[:, 0, 1]))
            print('Arousal MSE = {0}\n'.format(arousal_mse))

            # calculate PCC
            valence_pcc = np.append(valence_pcc, compute_pcc(predicted[:, 0], y_test[:, 0, 0]))
            print('Valence PCC = {0}\n'.format(valence_pcc))
            arousal_pcc = np.append(arousal_pcc, compute_pcc(predicted[:, 1], y_test[:, 0, 1]))
            print('Arousal PCC = {0}\n'.format(arousal_pcc))
        else:
            # calculate root mean squared error
            valence_mse = np.append(valence_mse, mean_squared_error(predicted[:, 0, 0], y_test[:, 0, 0]))
            print('Valence MSE = {0}\n'.format(valence_mse))
            arousal_mse = np.append(arousal_mse, mean_squared_error(predicted[:, 0, 1], y_test[:, 0, 1]))
            print('Arousal MSE = {0}\n'.format(arousal_mse))

            # calculate PCC
            valence_pcc = np.append(valence_pcc, compute_pcc(predicted[:, 0, 0], y_test[:, 0, 0]))
            print('Valence PCC = {0}\n'.format(valence_pcc))
            arousal_pcc = np.append(arousal_pcc, compute_pcc(predicted[:, 0, 1], y_test[:, 0, 1]))
            print('Arousal PCC = {0}\n'.format(arousal_pcc))

    scores = [np.mean(valence_mse), np.mean(arousal_mse), np.mean(valence_pcc), np.mean(arousal_pcc)]

    return scores, fig_path


def train(num_epochs, cells, opt, bs, ts, dp, lr, split=0):

    # Start Experiment
    e = Experiment(num_epochs, cells, opt, bs, ts, dp, lr, split)
    Bot.send_message('Starting experiment {0}...'.format(e.experiment_id))

    # Experimenting ...
    scores, fig_path = train_and_evaluate_model(e.experiment_id, num_epochs, cells, opt, bs,
                                                ts, dp, lr, split)

    # Save Experiment
    e.save_results(scores)
    Bot.send_image(fig_path)


if __name__ == '__main__':
    epochs = 1
    lstm_cells = 10
    optimizer = 'Adam'
    batch_size = 1
    timesteps = 1
    dropout = 0.5
    data_split = 0
    # train(epochs, lstm_cells, optimizer, batch_size, timesteps, dropout, data_split=data_split)
    train(100, 1024, None, 1, 1, 0.5, 1e-5, split=0)
