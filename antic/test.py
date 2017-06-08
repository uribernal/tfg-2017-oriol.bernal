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
from helper import TelegramBot as Bot
import os
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model
np.set_printoptions(threshold=np.nan)

def model_valence_arousal(batch_size=1, time_step = 1, dropout_probability=0.5, summary=False):
    weights_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/Mixed_features_e_0_b001_d0.5.hdf5'
    input_features = Input(batch_shape=(batch_size, time_step, 7168,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=False, name='lsmt1')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    output = TimeDistributed(Dense(2, activation='tanh'), name='fc')(output_dropout)

    model = Model(input=input_features, output=output)
    model.load_weights(weights_path)
    if summary:
        model.summary()
    return model


movies = Dm.get_movies_names()
path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/training_feat.h5'

movie = movies[1]
with h5py.File(path, 'r') as hdf:
    labels = np.array(hdf.get('dev/labels/' + movie))
    features = np.array(hdf.get('dev/features/' + movie))
len = features.shape[0] * 7 // 10
labels = labels.reshape(labels.shape[0], 1, labels.shape[1])
features = features.reshape(features.shape[0], 1, features.shape[1])

x = features[:]
y = labels[:, :, :2]

model = model_valence_arousal()
pred = model.predict(x)
print(pred.shape)
print(np.sum(pred[:,:,0]-y[:,:,0]))
for i in range(pred.shape[0]):
    print('{0}  {1}'.format(y[i,:,0], pred[i, :, 0]))