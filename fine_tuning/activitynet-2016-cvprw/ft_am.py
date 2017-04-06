#https://github.com/imatge-upc/activitynet-2016-cvprw/blob/master/scripts/train.py

import argparse
import os
import sys

import h5py

from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop

'''
Use of C3D model to extract features for the videos
    Split the videos in 16 frames and resize them to 178x128 to fit the input of the C3D model
'''

experiment_id = 1
input_dataset = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/fine_tuning/activitynet-2016-cvprw/c3d-sports1M_weights.h5'
num_cells = 512
num_layers = 1
dropout_probability = 0.5
batch_size = 256
timesteps = 20
epochs = 100
lr = 1e-5
loss_weight = 0.3

store_weights_root = 'data/model_snapshot'
store_weights_file = 'lstm_activity_classification_{experiment_id}_e{epoch:03}.hdf5'

print('Compiling model')
input_features = Input(batch_shape=(batch_size, timesteps, 4096,), name='features')
input_normalized = BatchNormalization(name='normalization')(input_features)
input_dropout = Dropout(p=dropout_probability)(input_normalized)
lstms_inputs = [input_dropout]
for i in range(num_layers):
    previous_layer = lstms_inputs[-1]
    lstm = LSTM(num_cells, return_sequences=True, stateful=True, name='lsmt{}'.format(i+1))(previous_layer)
    lstms_inputs.append(lstm)

output_dropout = Dropout(p=dropout_probability)(lstms_inputs[-1])
output = TimeDistributed(Dense(201, activation='softmax'), name='fc')(output_dropout)

model = Model(input=input_features, output=output)
model.summary()
rmsprop = RMSprop(lr=lr)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'], sample_weight_mode='temporal')
print('Model Compiled!')

print('Loading Training Data...')
f_dataset = h5py.File(input_dataset, 'r')

X = f_dataset['training']['vid_features']
Y = f_dataset['training']['output']

print(X)
print(Y)

print('Loading Sample Weights...')
sample_weight = f_dataset['training']['sample_weight'][...]
sample_weight[sample_weight != 1] = loss_weight
print('Loading Validation Data...')
X_val = f_dataset['validation']['vid_features']
Y_val = f_dataset['validation']['output']

print('Loading Data Finished!')
print('Input shape: {}'.format(X.shape))
print('Output shape: {}\n'.format(Y.shape))
print('Validation Input shape: {}'.format(X_val.shape))
print('Validation Output shape: {}'.format(Y_val.shape))
print('Sample Weights shape: {}'.format(sample_weight.shape))

for i in range(1, epochs+1):
    print('Epoch {}/{}'.format(i, epochs))
    model.fit(X,
            Y,
            batch_size=batch_size,
            validation_data=(X_val, Y_val),
            sample_weight=sample_weight,
            verbose=1,
            nb_epoch=1,
            shuffle=False)
    print('Reseting model states')
    model.reset_states()
    if (i % 5) == 0:
        print('Saving snapshot...')
        save_name = store_weights_file.format(experiment_id=experiment_id, epoch=i)
        save_path = os.path.join(store_weights_root, save_name)
        model.save_weights(save_path)
