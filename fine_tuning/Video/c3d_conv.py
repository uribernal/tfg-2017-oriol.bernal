import numpy as np
from helper import VideoHelper as Vh
from helper import ModelGenerator as Mg


def get_C3D_features(videos, print_info=False):
    from helper.DatasetManager import videos_path, videos_extension
    c3d_features = np.array([])
    for video in videos:
        input_video = videos_path + video + videos_extension
        input_size = (112, 112)
        length = 16
        if print_info:
            print('Reading Video...')
        video_array = Vh.video_to_array(input_video, resize=input_size)
        if video_array is None:
            raise Exception('The video could not be read')
        nb_frames = Vh.get_num_frames(input_video)
        duration = Vh.get_duration(input_video)
        fps = nb_frames / duration
        if print_info:
            print('Duration: {:.1f}s'.format(duration))
            print('FPS: {:.1f}'.format(fps))
            print('Number of frames: {}'.format(nb_frames))

        nb_clips = nb_frames // length
        video_array = video_array.transpose(1, 0, 2, 3)
        video_array = video_array[:nb_clips*length,:,:,:]
        video_array = video_array.reshape((nb_clips, length, 3, 112, 112))
        video_array = video_array.transpose(0, 2, 1, 3, 4)

        # Load C3D model and mean
        if print_info:
            print('Loading C3D network...')
        model = C3D_conv_features(True)
        model.compile(optimizer='sgd', loss='mse')
        mean_total = np.load('data/models/c3d-sports1M_mean.npy')
        mean = np.mean(mean_total, axis=(0, 2, 3, 4), keepdims=True)

        # Extract features
        if print_info:
            print('Extracting features...')
        X = video_array - mean
        Y = model.predict(X, batch_size=1, verbose=1)
        print(Y.shape)
        c3d_features = np.append(c3d_features, Y)
    print(c3d_features)
    return (c3d_features)

def C3D_conv_features(summary=False):
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
    model.load_weights('data/models/c3d-sports1M_weights.h5')

    for _ in range(4):
        model.pop_layer()

    if summary:
        print(model.summary())
    return model