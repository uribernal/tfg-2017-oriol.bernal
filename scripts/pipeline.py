import numpy as np
from scripts.extract_audios import extract_audio
from scripts.videos_2_30fps import video_2_30fps
from scripts.extract_video_features import extract_video_features
from scripts.extract_audio_features import extract_audio_features


def get_model(summary=False):

    from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
    from keras.models import Model
    input_features = Input(batch_shape=(1, 1, 7168,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(0.5)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=False, name='lsmt1')(input_dropout)
    output_dropout = Dropout(0.5)(lstm)
    output = TimeDistributed(Dense(2, activation='tanh'), name='fc')(output_dropout)

    model = Model(input=input_features, output=output)
    model.load_weights('/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/train_with_Mix_features_e_1_b064_d0.5.hdf5')
    if summary:
        model.summary()
    '''
    import keras
    model = keras.models.load_model('/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/train_with_Mix_features_e_1_b064_d0.5.hdf5')
     '''
    return model

video_input = '/home/uribernal/Desktop/data/dev/movies/'
video_30fps_path = '/home/uribernal/Desktop/data/dev/resized_movies/'
audio_path = '/home/uribernal/Desktop/data/dev/audios/'
# video_name = 'MEDIAEVAL16_00000'
video_name = 'On_time'

video_extension = '.mp4'
audio_extension = '.wav'

if False:
    # EXTRACT AUDIO FROM VIDEO FILE
    input_vid = video_input + video_name + video_extension
    output_vid = audio_path + video_name + audio_extension
    extract_audio(input_vid, output_vid)

    # CONVERT VIDEO TO 30 FPS
    video_shape = (112, 112)
    output_vid = video_30fps_path + video_name + video_extension
    video_2_30fps(input_vid, output_vid, video_shape=video_shape)

    # EXTRACT VIDEO FEATURES
    video_features = extract_video_features(video_name, video_30fps_path)
    print('Video features: {0}\n'.format(video_features.shape))


    # EXTRACT AUDIO FEATURES
    audio_features = extract_audio_features(video_name, audio_path)
    print('Audio features: {0}\n'.format(audio_features.shape))

    # FUSE FEATURES BY CONCATENATION
    features = np.append(video_features, audio_features, axis=1)
    num_feat = video_features.shape[1] + audio_features.shape[1]
    print('Features: {0}\n'.format(features.shape))

# PREDICT
model = get_model(summary=True)

# SHOW PREDICTIONS

