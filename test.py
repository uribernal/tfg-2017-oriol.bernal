import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model
import numpy as np
import h5py
from helper import TelegramBot as Bot


figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/test/Test.png'

data_path = '/home/uribernal/Desktop/MediaEval2017/data/data/emotional_impact.h5'

with h5py.File(data_path, 'r') as hdf:
    labels = np.array(hdf.get('labels'))
    data_b = np.array(hdf.get('audio_features'))
data_b = data_b.reshape(data_b.shape[0], 1, data_b.shape[1] * data_b.shape[2] * data_b.shape[3])
with h5py.File('/home/uribernal/Desktop/MediaEval2017/data/data/provisional_video_features.h5', 'r') as hdf:
    data_a = np.array(hdf.get('features'))
data_a = data_a.reshape(data_b.shape[0], 1, data_a.shape[1])
data_a = data_a[:, :, :3072]
labels = labels[:, :2]

print('data_a shape: {}'.format(data_a.shape))
print('data_b shape: {}'.format(data_b.shape))
print('Labels shape: {}'.format(labels.shape))

tweet_a = Input(shape=(None, 3072))
tweet_b = Input(shape=(None, 3072))

# This layer can take as input a matrix
# and will return a vector of size 64
shared_lstm = LSTM(512)

# When we reuse the same layer instance
# multiple times, the weights of the layer
# are also being reused
# (it is effectively *the same* layer)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# We can then concatenate the two vectors:
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# And add a logistic regression on top
predictions = Dense(2, activation='tanh')(merged_vector)

# We define a trainable model linking the
# tweet inputs to the predictions
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='mean_squared_error')

# Training
train_loss = []
validation_loss = []
history = model.fit([data_a, data_b], labels, epochs=10)

train_loss.extend(history.history['loss'])
validation_loss.extend(history.history['val_loss'])

Bot.save_plots(train_loss, validation_loss, figures_path)