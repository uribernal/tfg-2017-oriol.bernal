# LSTM for international airline passengers problem with time step regression framing
import h5py
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from helper import TelegramBot as Bot
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model


def get_callbacks(model_checkpoint, patience1, patience2):
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

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


movies = ['After_The_Rain', 'Attitude_Matters', 'Barely_legal_stories', 'Between_Viewings', 'Big_Buck_Bunny', 'Chatter', 'Cloudland', 'Damaged_Kung_Fu', 'Decay', 'Elephant_s_Dream', 'First_Bite', 'Full_Service', 'Islands', 'Lesson_Learned', 'Norm', 'Nuclear_Family', 'On_time', 'Origami', 'Parafundit', 'Payload', 'Riding_The_Rails', 'Sintel', 'Spaceman', 'Superhero', 'Tears_of_Steel', 'The_room_of_franz_kafka', 'The_secret_number', 'To_Claire_From_Sonny', 'Wanted', 'You_Again']

db_path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/training_feat.h5'
l = numpy.array([])
f = numpy.array([])
#movies = [movies[22]]
for movie in movies:
    with h5py.File(db_path, 'r') as hdf:
        labels = numpy.array(hdf.get('dev/labels/' + movie))
        features = numpy.array(hdf.get('dev/features/' + movie))
    l = numpy.append(l, labels)
    f = numpy.append(f, features)
labels = l.reshape(l.shape[0]//3, 1, 3)
features = f.reshape(f.shape[0]//7168, 1, 7168)
print('Labels´ shape: {}'.format(labels.shape))
print('Features´ shape: {}\n'.format(features.shape))

# split into train and test sets
train_size = int(labels.shape[0] * 0.67)
test_size = labels.shape[0] - train_size
train_x, test_x = features[0:train_size, :, :], features[train_size:labels.shape[0], :, :]
train_y, test_y = labels[0:train_size, :, 0], labels[train_size:labels.shape[0], :, 0]
print('X_train shape: {}'.format(train_x.shape))
print('Y_train shape: {}'.format(train_y.shape))
print('X_test shape: {}'.format(test_x.shape))
print('Y_test shape: {}\n'.format(test_y.shape))

# Path for the weights
store_weights_file = 'Mixed_features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.hdf5'

model_checkpoint = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/' + \
                       store_weights_file.format(experiment_id=0, batch_size=1,
                                                 drop_out=0)
lr_patience = 10  # When to decrease lr
stop_patience = 80  # When to finish trainning if no learning
callbacks = get_callbacks(model_checkpoint, lr_patience, stop_patience)

# create and fit the LSTM network
look_back = 1
#model = Sequential()
#model.add(LSTM(4, input_shape=(1, 7168)))
#model.add(Dense(1))

input_features = Input(batch_shape=(1, 1, 7168,), name='features')
input_normalized = BatchNormalization(name='normalization')(input_features)
input_dropout = Dropout(0.5)(input_normalized)
lstm = LSTM(512, return_sequences=True, stateful=False, name='lsmt1')(input_dropout)
output_dropout = Dropout(0.5)(lstm)
output = TimeDistributed(Dense(2, activation='tanh'), name='fc')(output_dropout)

model = Model(input=input_features, output=output)


model.compile(loss='mean_squared_error', optimizer='adam')
train_loss = []
validation_loss = []
history = model.fit(train_x, train_y, nb_epoch=10, batch_size=1, verbose=2)
train_loss.extend(history.history['loss'])
#validation_loss.extend(history.history['val_loss'])
# Path for the figures
figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/mixed_features/' + \
               '{min:05}_Mixed_Features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.png'


# make predictions
trainPredict = model.predict(train_x)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(train_y[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))

testPredict = model.predict(test_x)

testScore = math.sqrt(mean_squared_error(test_y[:, 0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

Bot.save_plots(train_loss, validation_loss, figures_path.format(min=1, experiment_id=0,
                                                                batch_size=1,
                                                                drop_out=0))