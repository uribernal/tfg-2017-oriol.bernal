"""
This assistant returns different Keras models (architectures)
that may need to be compiled and trained.
"""

from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model


def c3d_conv_features(summary=False):
    """ Return the Keras model of the network until the fc6 layer where the
    convolutional features can be extracted.
    """
    from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Sequential

    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1),
                            input_shape=(3, 16, 112, 112),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5a',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5b',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropadding'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    model.add(Flatten(name='flatten'))
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6', trainable=False))
    model.add(Dropout(.5, name='do1'))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5, name='do2'))
    model.add(Dense(487, activation='softmax', name='fc8'))

    # Load weights
    model.load_weights('/home/uribernal/Downloads/activitynet-2016-cvprw-master/data/models/c3d-sports1M_weights.h5')

    for _ in range(4):
        model.pop_layer()

    if summary:
        print(model.summary())
    return model


def lstm_alberto_tfg_c3d(batch_size=32, time_steps=1, dropout_probability=.5, summary=False):

    input_features = Input(batch_shape=(batch_size, time_steps, 4096,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=False, name='lsmt1')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    output = TimeDistributed(Dense(1, activation='tanh'), name='fc')(output_dropout)

    model = Model(inputs=input_features, outputs=output)

    if summary:
        model.summary()
    return model


def lstm_alberto_tfg_activities(batch_size=32, dropout_probability=.5, summary=False):

    input_features = Input(batch_shape=(batch_size, 1, 201,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=False, name='lsmt1')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    output = TimeDistributed(Dense(1, activation='tanh'), name='fc')(output_dropout)

    model = Model(inputs=input_features, outputs=output)

    if summary:
        model.summary()
    return model


def three_layers_lstm(s1=2048, s2=1024, s3=512, batch_size=32, time_steps=1, dropout_probability=.5, summary=False):

    input_features = Input(batch_shape=(batch_size, time_steps, 4096), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(s1, return_sequences=True, stateful=True, name='lsmt1')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    lstm2 = LSTM(s2, return_sequences=True, stateful=True, name='lsmt2')(output_dropout)
    output_dropout2 = Dropout(dropout_probability)(lstm2)
    lstm3 = LSTM(s3, return_sequences=True, stateful=True, name='lsmt3')(output_dropout2)
    output_dropout3 = Dropout(dropout_probability)(lstm3)
    output = TimeDistributed(Dense(1, activation='tanh'), name='fc')(output_dropout3)

    model = Model(inputs=input_features, outputs=output)

    if summary:
        model.summary()
    return model


def two_layers_lstm(s1=2048, s3=512, batch_size=32, time_steps=1, dropout_probability=.5, summary=False):

    input_features = Input(batch_shape=(batch_size, time_steps, 4096), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(s1, return_sequences=True, stateful=True, name='lsmt1')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    output_dropout2 = Dropout(dropout_probability)(output_dropout)
    lstm2 = LSTM(s3, return_sequences=True, stateful=True, name='lsmt3')(output_dropout2)
    output_dropout3 = Dropout(dropout_probability)(lstm2)
    output = TimeDistributed(Dense(1, activation='tanh'), name='fc')(output_dropout3)

    model = Model(inputs=input_features, outputs=output)

    if summary:
        model.summary()
    return model


def lstm_audio(batch_size=32, dropout_probability=.5, summary=False):

    input_features = Input(batch_shape=(batch_size, 1, 98*64), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(dropout_probability)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=False, name='lsmt')(input_dropout)
    output_dropout = Dropout(dropout_probability)(lstm)
    output = TimeDistributed(Dense(1, activation='tanh'), name='fc')(output_dropout)

    model = Model(inputs=input_features, outputs=output)

    if summary:
        model.summary()
    return model


def alex_net(summary=False):
    docs = 'https://gist.github.com/JBed/c2fb3ce8ed299f197eff'

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization

    # AlexNet with batch normalization in Keras
    # input image is 224x224

    model = Sequential()
    model.add(Convolution2D(64, 3, 11, 11, border_mode='full'))
    model.add(BatchNormalization((64, 226, 226)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution2D(128, 64, 7, 7, border_mode='full'))
    model.add(BatchNormalization((128, 115, 115)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution2D(192, 128, 3, 3, border_mode='full'))
    model.add(BatchNormalization((128, 112, 112)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution2D(256, 192, 3, 3, border_mode='full'))
    model.add(BatchNormalization((128, 108, 108)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Flatten())
    model.add(Dense(12 * 12 * 256, 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096, 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096, 1, init='normal'))
    model.add(BatchNormalization(1000))
    model.add(Activation('tanh'))

    if summary:
        model.summary()
    return model


def vgg_16(batch_size=32, summary=False, weights_path=None):
    docs = 'https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3'

    from keras.models import Sequential
    from keras.layers.core import Flatten, Dense, Dropout
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(batch_size, 98, 64)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    #model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    if summary:
        model.summary()
    return model


def c3d_audio(batch_size=32, summary=False):
    """ Return the Keras model of the network until the fc6 layer where the
    convolutional features can be extracted.
    """
    from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Sequential

    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(64, (3, 3, 3), activation='relu',
                            padding='same', name='conv1',
                            strides=(1, 1, 1),
                            input_shape=(98, 64),
                            trainable=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, (3, 3, 3), activation='relu',
                            padding='same', name='conv2',
                            strides=(1, 1, 1),
                            trainable=True))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3a',
                            strides=(1, 1, 1),
                            trainable=True))
    model.add(Convolution3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3b',
                            strides=(1, 1, 1),
                            trainable=True))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv4a',
                            strides=(1, 1, 1),
                            trainable=True))
    model.add(Convolution3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv4b',
                            strides=(1, 1, 1),
                            trainable=True))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv5a',
                            strides=(1, 1, 1),
                            trainable=True))
    model.add(Convolution3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv5b',
                            strides=(1, 1, 1),
                            trainable=True))
    model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropadding'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))
#    model.add(Flatten(name='flatten'))
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6', trainable=True))
    model.add(Dropout(.5, name='do1'))
    model.add(Dense(4096, activation='relu', name='fc7'))

    if summary:
        print(model.summary())
    return model
