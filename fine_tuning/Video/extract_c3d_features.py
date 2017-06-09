import os.path
import numpy as np
import h5py
from helper import VideoHelper as Vh


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


def extract_features(video_name, video_path=None, video_extension=None, features_path=None, saved_data=False,
                     input_size=(112, 112), length=16):

    if features_path is not None:
        # If file doesn't exist, create it
        if not os.path.isfile(features_path):
            # Create the HDF5 file
            hdf = h5py.File(features_path, 'w')
            hdf.close()

    if saved_data:
        from helper.DatasetManager import data_path
        with h5py.File(data_path, 'r') as hdf:
            video_array = np.array(hdf.get('dev/visual_data/' + video_name))
    else:
        if video_path is None:
            from helper.DatasetManager import videos_path as video_path
        if video_extension is None:
            from helper.DatasetManager import videos_extension as video_extension

        video_array = Vh.get_resized_video(video_name, video_path=video_path, video_extension = video_extension, print_info=True)


    print('Loading C3D network...')
    model = C3D_conv_features(length, input_size, summary=False)
    model.compile(optimizer='sgd', loss='mse')
    mean_total = np.load(
        '/home/uribernal/Downloads/activitynet-2016-cvprw-master/data/models/c3d-sports1M_mean.npy')
    mean = np.mean(mean_total, axis=(0, 2, 3, 4), keepdims=True)

    print('Extracting features...')
    x = video_array - mean
    predictions = model.predict(x, batch_size=1, verbose=0)

    # If h5py file path is given, save features in the h5py file
    if features_path is not None:
        with h5py.File(features_path, 'r+') as hdf:
            hdf.create_dataset('dev/visual_features/' + video_name, data=predictions, compression='gzip', compression_opts=9)
        print('{} stored'.format(video_name))

    return predictions