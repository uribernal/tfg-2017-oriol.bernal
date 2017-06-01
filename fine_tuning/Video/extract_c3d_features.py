from helper import DatasetManager as Dm
from helper import ModelGenerator as Mg
from helper import VideoHelper as Vh
import numpy as np
import h5py


def C3D_conv_features(length, input_size, summary=False):
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
                            input_shape=(3, length, input_size[0], input_size[1]),
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
    model.load_weights(
        '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/fine_tuning/Video/c3d-sports1M_weights.h5')

    for _ in range(4):
        # model.pop_layer()
        model.pop()

    if summary:
        print(model.summary())
    return model


movies = Dm.get_movies_names()
input_size = (112, 112)
length = 16
del movies[22]
for movie in movies:
    with h5py.File('/home/uribernal/Desktop/MediaEval2017/data/data/data/emotional_impact.h5', 'r') as hdf:
        labels = np.array(hdf.get('dev/ground_truth_data/' + movie))
        video_array = np.array(hdf.get('dev/visual_data/' + movie))
    print(video_array.shape)
    lab = labels.shape[0]
    fps = 30
    # Load C3D model and mean
    print('Loading C3D network...')
    model = C3D_conv_features(length, input_size, False)
    model.compile(optimizer='sgd', loss='mse')
    mean_total = np.load('/home/uribernal/Downloads/activitynet-2016-cvprw-master/data/models/c3d-sports1M_mean.npy')
    mean = np.mean(mean_total, axis=(0, 2, 3, 4), keepdims=True)

    # Extract features
    print('Extracting features...')
    X = video_array - mean
    Y = model.predict(X, batch_size=1, verbose=0)
    #######################################
    #res = Y.shape[0] % lab
    #len = Y.shape[0] - res
    #Y = Y[:len, :]
    #k = int(np.floor(5 * fps / 16))
    #Y1 = Y.reshape(int(Y.shape[0] / k), k, Y.shape[1])
    #Y2 = np.sum(Y1, axis=1) / 9

    path_file = '/home/uribernal/Desktop/MediaEval2017/data/data/data/emotional_impact.h5'

    with h5py.File(path_file, 'r+') as hdf:
        hdf.create_dataset('dev/visual_features/' + movie, data=Y, compression='gzip', compression_opts=9)
    print('{} stored'.format(movie))
print('Finished')

