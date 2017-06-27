import numpy as np
import h5py
from helper import DatasetManager as Dm
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from helper import TelegramBot as Bot
from sklearn.metrics import mean_squared_error
from helper.DatasetManager import compute_pcc
from helper.ExperimentHelper import Experiment
import time


np.set_printoptions(threshold=np.NaN)
figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/video_features/' + \
               'Video_Features_e_{experiment_id}_b{batch_size:03}_c{cells}.png'

store_weights_file = 'Video_features_e_{experiment_id}_b{batch_size:03}_c{cells}.hdf5'

db_path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/training_feat.h5'


def get_model(cells=512, bs=1, ts=1, i_dim=4096, dp=0.5, opt=None, summary=False):
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


def get_data(split):
    # Get data Train, Validation and Test
    movies = Dm.get_movies_names()

    movies_val = movies[split + 1]
    movies_test = movies[split]
    del movies[split]
    del movies[split]
    movies_train = movies

    with h5py.File(db_path, 'r') as hdf:
        labels = np.array(hdf.get('dev/labels/' + movies_val))
        features = np.array(hdf.get('dev/features/' + movies_val))
    labels_val = labels.reshape(labels.shape[0], 1, labels.shape[1])[:, :, :2]
    features_val = features.reshape(features.shape[0], 1, features.shape[1])[:, :, :4096]
    with h5py.File(db_path, 'r') as hdf:
        labels = np.array(hdf.get('dev/labels/' + movies_test))
        features = np.array(hdf.get('dev/features/' + movies_test))
    labels_test = labels.reshape(labels.shape[0], 1, labels.shape[1])[:, :, :2]
    features_test = features.reshape(features.shape[0], 1, features.shape[1])[:, :, :4096]

    return movies_train, features_val, labels_val, features_test, labels_test


def train_and_evaluate_model(experiment_id, num_epochs, cells, opt, bs, ts, dp, split):

    # Path for results
    fig_path = figures_path.format(experiment_id=experiment_id, batch_size=bs, cells=cells)

    # Get the model
    model = get_model(cells=cells, bs=bs, ts=ts, dp=dp, opt=opt, summary=True)

    # Get the data
    movies_train, x_val, y_val, x_test, y_test = get_data(split)

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
            x_train = features.reshape(features.shape[0], 1, features.shape[1])[:, :, :4096]
            for i in range(len(x_train)//bs):
                print(len(x_train))
                a = len(x_train) // bs
                k = len(x_train) % bs
                print(a)
                print(k)
                if i == a:
                    x = x_train[i * bs:(i + 1) * k].reshape(k, 1, 4096)
                    y_true = y_train[i * bs:(i + 1) * k, 0, :]
                    history = model.fit(x, y_true, batch_size=k, nb_epoch=1)

                else:
                    x = x_train[i*bs:(i+1)*bs].reshape(bs, 1, 4096)
                    y_true = y_train[i*bs:(i+1)*bs, 0, :]
                    history = model.fit(x, y_true, batch_size=64, nb_epoch=1)
                tr_loss_movies.append(history.history['loss'])
            model.reset_states()

        te_loss_movies = []
        for i in range(len(x_val)):
            x = np.expand_dims(np.expand_dims(x_val[i][0], axis=0), axis=0)
            y_true = np.array([y_val[i, 0, :]])
            pred_val = model.predict(x, batch_size=1)
            te_loss_movies.append(te_loss)
        model.reset_states()

        training_loss_epochs = np.append(training_loss_epochs, np.mean(tr_loss_movies))
        test_loss_epochs = np.append(test_loss_epochs, np.mean(te_loss_movies))

        print('epoch = {}, loss training = {}, loss testing = {}, elapsed time = {}'
              ''.format(epoch, np.mean(tr_loss_movies), np.mean(te_loss_movies), time.time()-start))
        print('___________________________________')

    # Save results
    Bot.save_plots(training_loss_epochs, test_loss_epochs, fig_path)
    json_file = model.to_json()
    with open('/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/models/model_{}.json'.format(
                experiment_id), 'w') as f:
        f.write(json_file)

    model.save_weights('/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/models/weights_{}.json'.format(
                experiment_id))

    # Predict with de TEST set
    predicted = model.predict(x_test, batch_size=1)
    print(predicted)
    print(predicted.shape)
    # calculate root mean squared error
    valence_mse = mean_squared_error(predicted[:, 0, 0], y_test[:, 0, 0])
    print('Valence MSE = {0}\n'.format(valence_mse))
    arousal_mse = mean_squared_error(predicted[:, 0, 1], y_test[:, 0, 1])
    print('Arousal MSE = {0}\n'.format(arousal_mse))

    # calculate PCC
    valence_pcc = compute_pcc(predicted[:, 0], y_test[:, 0, 0])
    print('Valence PCC = {0}\n'.format(valence_pcc))
    arousal_pcc = compute_pcc(predicted[:, 1], y_test[:, 0, 1])
    print('Arousal PCC = {0}\n'.format(arousal_pcc))

    scores = [np.mean(valence_mse), np.mean(arousal_mse), np.mean(valence_pcc), np.mean(arousal_pcc)]

    return scores, fig_path


def train(num_epochs, cells, opt, bs, ts, dp, split=0):

    # Start Experiment
    e = Experiment(num_epochs, cells, opt, bs, ts, dp, split)
    Bot.send_message('Starting experiment {0}...'.format(e.experiment_id))

    # Experimenting ...
    scores, fig_path = train_and_evaluate_model(e.experiment_id, num_epochs, cells, opt, bs,
                                                ts, dp, split)

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

    train(1, 1, 'Adam', 64, 1, 0.5, split=0)
