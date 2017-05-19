import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.layers import TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
import numpy as np

import h5py
'''
figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/test/Test.png'
data_path = '/home/uribernal/Desktop/MediaEval2017/data/data/emotional_impact.h5'

with h5py.File(data_path, 'r') as hdf:
    labels = np.array(hdf.get('labels'))
    data_b = np.array(hdf.get('audio_features'))
    data_a = np.array(hdf.get('features'))

data_b = data_b.reshape(data_b.shape[0], 1, data_b.shape[1] * data_b.shape[2] * data_b.shape[3])
data_a = data_a.reshape(data_b.shape[0], 1, data_a.shape[1])
labels = labels[:, :2]
print('data_a shape: {}'.format(data_a.shape))
print('data_b shape: {}'.format(data_b.shape))
print('Labels shape: {}'.format(labels.shape))
'''


#################################################################################################
# First, let's define a vision model using a Sequential model.
# This model will encode an image into a vector.
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(3, 224, 224)))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())

# Next, let's define a language model to encode the question into a vector.
# Each question will be at most 100 word long,
# and we will index words as integers from 1 to 9999.
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)



###################################################################################################
video_input = Input(shape=(None, 3, 224, 224))
# This is our video encoded via the previously trained vision_model (weights are reused)
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # the output will be a sequence of vectors
encoded_video = LSTM(512)(encoded_frame_sequence)  # the output will be a vector

# This is a model-level representation of the question encoder, reusing the same weights as before:
question_encoder = Model(inputs=question_input, outputs=encoded_question)

# Let's use it to encode the question:
video_question_input = Input(shape=(3072,), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

# And this is our video question answering model:
merged = keras.layers.concatenate([encoded_video, encoded_video_question])
output = Dense(2, activation='tanh')(merged)

video_qa_model = Model(inputs=[video_input,  video_question_input], outputs=output)
###################################################################################################

video_qa_model.summary()
