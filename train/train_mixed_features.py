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
from helper.DatasetManager import compress_labels
import time


# When printing, all the samples will be shown
np.set_printoptions(threshold=np.NaN)

# Constant PATHS for the script
db_path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/training_feat.h5'
model_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/models/model_{}.h5'
weights_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/models/weights_{}.h5'
figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/mixed_features/' + \
                   'Mixed_Features_e_{experiment_id}_b{batch_size:03}_c{cells}.png'

# Boolean for sending information to mobile phone (using telegram)
use_telegram_bot = True


def get_model(cells=512, bs=1, ts=1, i_dim=7168, dp=0.5, opt=None, lr=1e-8, summary=False):
    """ Returns a model composed with an LSTM """

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
    """ Returns a partition of the DB (train, validation and Test) """

    # Get all the movies from the develpment set
    movies = Dm.get_movies_names()

    # Get validation films
    movies_val = [movies[split], movies[split + 1], movies[split + 2]]

    # Get test films
    movies_test = [movies[split + 3], movies[split + 4]]

    # Get train films
    del movies[split]
    del movies[split]
    del movies[split]
    del movies[split]
    del movies[split]
    movies_train = movies

    return movies_train, movies_val, movies_test


def train_and_evaluate_model(experiment_id, num_epochs, cells, opt, bs, ts, dp, lr, split):
    """ Trains the model and evaluates its performance
    The weights and the model can be found at weights_path and model_path defined above
    The training and validation curves can be found at figures_path defined above """

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

        # TRAINING
        for movie in movies_train:

            # Get inputs (features and ground truth data for a single movie)
            with h5py.File(db_path, 'r') as hdf:
                labels = np.array(hdf.get('dev/labels/' + movie))
                features = np.array(hdf.get('dev/features/' + movie))
            y_train = labels.reshape(labels.shape[0], 1, labels.shape[1])[:, :, :2]
            x_train = features.reshape(features.shape[0], 1, features.shape[1])[:, :, :]

            # For every feature vector -> train on batch
            for i in range(len(x_train)):
                x = np.expand_dims(np.expand_dims(x_train[i][0], axis=0), axis=0)
                y_true = np.array([y_train[i, :, :]])
                tr_loss = model.train_on_batch(x, y_true)
                tr_loss_movies.append(tr_loss)

            # After the network has seen all the video inputs, reset the states to start again with another video
            model.reset_states()

        # VALIDATION
        val_loss_movies = []
        for movie in movies_val:

            # Get inputs (features and ground truth data for a single movie)
            with h5py.File(db_path, 'r') as hdf:
                labels = np.array(hdf.get('dev/labels/' + movie))
                features = np.array(hdf.get('dev/features/' + movie))
            y_val = labels.reshape(labels.shape[0], 1, labels.shape[1])[:, :, :2]
            x_val = features.reshape(features.shape[0], 1, features.shape[1])[:, :, :]

            # For every feature vector -> test on batch
            for i in range(len(x_val)):
                x = np.expand_dims(np.expand_dims(x_val[i][0], axis=0), axis=0)
                y_true = np.array([y_val[i, :, :]])
                val_loss = model.test_on_batch(x, y_true)
                val_loss_movies.append(val_loss)
            # After the network has seen all the video inputs, reset the states to start again with another video
            model.reset_states()

        # Get the mean average of all the movies
        training_loss_epochs = np.append(training_loss_epochs, np.mean(tr_loss_movies))
        test_loss_epochs = np.append(test_loss_epochs, np.mean(val_loss_movies))

        print('epoch = {}, loss training = {}, loss testing = {}, elapsed time = {}'
              .format(epoch, np.mean(tr_loss_movies), np.mean(val_loss_movies), time.time()-start))
        print('___________________________________')

        # Send results to mobile phone
        if use_telegram_bot:
            Bot.send_message('epoch = {}, loss training = {}, loss testing = {}, elapsed time = {}'.
                             format(epoch, np.mean(tr_loss_movies), np.mean(val_loss_movies), time.time() - start))

    # Save plot (training and validation curves)
    Dm.save_plots(training_loss_epochs, test_loss_epochs, fig_path)

    # Save model (physical structure)
    model.save(model_path.format(experiment_id))

    # Save weights
    model.save_weights(weights_path.format(
                experiment_id))

    # TEST
    valence_mse = np.array([])
    arousal_mse = np.array([])
    valence_pcc = np.array([])
    arousal_pcc = np.array([])

    # Predict with de TEST set
    for movie in movies_test:

        # Get inputs (features and ground truth data for a single movie)
        with h5py.File(db_path, 'r') as hdf:
            labels = np.array(hdf.get('dev/labels/' + movie))
            features = np.array(hdf.get('dev/features/' + movie))
        y_test = labels.reshape(labels.shape[0], 1, labels.shape[1])[:, :, :2]
        x_test = features.reshape(features.shape[0], 1, features.shape[1])[:, :, :]

        # Predict valence and arousal scores for a single movie
        predicted = model.predict(x_test, batch_size=1)

        # Print predictions
        print('----------------------PREDICTIONS-------------------------')
        print(predicted)

        # Print MSE and PCC scores
        # calculate root mean squared error
        valence_mse = np.append(valence_mse, mean_squared_error(compress_labels(predicted[:, 0, 0]), compress_labels(y_test[:, 0, 0])))
        print('Valence MSE = {0}\n'.format(valence_mse))
        arousal_mse = np.append(arousal_mse, mean_squared_error(compress_labels(predicted[:, 0, 1]), compress_labels(y_test[:, 0, 1])))
        print('Arousal MSE = {0}\n'.format(arousal_mse))

        # calculate PCC
        valence_pcc = np.append(valence_pcc, compute_pcc(compress_labels(predicted[:, 0, 0]), compress_labels(y_test[:, 0, 0])))
        print('Valence PCC = {0}\n'.format(valence_pcc))
        arousal_pcc = np.append(arousal_pcc, compute_pcc(compress_labels(predicted[:, 0, 1]), compress_labels(y_test[:, 0, 1])))
        print('Arousal PCC = {0}\n'.format(arousal_pcc))

        print('-----------------------------------------------')

    # Get mean scores (MSE and PCC) for valence and arousal
    scores = [np.mean(valence_mse), np.mean(arousal_mse), np.mean(valence_pcc), np.mean(arousal_pcc)]

    return scores, fig_path


def train(num_epochs, cells, opt, bs, ts, dp, lr, split=0):

    # Start Experiment
    e = Experiment(num_epochs, cells, opt, bs, ts, dp, lr, split)

    # Send information to mobile phone
    if use_telegram_bot:
        Bot.send_message('Starting experiment {0}...'.format(e.experiment_id))

    # Experimenting ...
    scores, fig_path = train_and_evaluate_model(e.experiment_id, num_epochs, cells, opt, bs, ts, dp, lr, split)

    # Save Experiment
    e.save_results(scores)

    # Send train and validation curves to mobile phone
    if use_telegram_bot:
        Bot.send_image(fig_path)


if __name__ == '__main__':
    epochs = 1
    lstm_cells = 10
    optimizer = 'Adam'
    batch_size = 1
    timesteps = 1
    dropout = 0.5
    data_split = 0

    # train(epochs, lstm_cells, optimizer, batch_size, timesteps, dropout, split=data_split)
    train(50, 1024, 'Adadelta', 1, 1, 0.5, 1e-5, split=5)
