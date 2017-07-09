import numpy as np
import h5py
from scripts.extract_audios import extract_audio
from scripts.videos_2_30fps import video_2_30fps
from scripts.extract_video_features import extract_video_features
from scripts.extract_audio_features import extract_audio_features
from sklearn.metrics import mean_squared_error
from helper.DatasetManager import compute_pcc
from helper.DatasetManager import compress_labels
from keras.models import load_model

# PATHS
model_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/models/model_11.h5'
weights_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/models/weights_11.h5'

video_input = '/home/uribernal/Desktop/data/dev/movies/'
video_30fps_path = '/home/uribernal/Desktop/data/dev/resized_movies/'
audio_path = '/home/uribernal/Desktop/data/dev/audios/'
# video_name = 'MEDIAEVAL16_00000'
video_name = 'On_time'
video_extension = '.mp4'
audio_extension = '.wav'

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
with h5py.File('/home/uribernal/Desktop/MediaEval2017/data/data/data/training_feat.h5', 'r') as hdf:
    y_test = np.array(hdf.get('dev/labels/' + video_name))
print('Video features: {0}\n'.format(video_features.shape))
print('Labels: {0}\n'.format(y_test.shape))

# EXTRACT AUDIO FEATURES
audio_features = extract_audio_features(video_name, audio_path)
print('Audio features: {0}\n'.format(audio_features.shape))

# FUSE FEATURES BY CONCATENATION
features = np.append(video_features, audio_features, axis=1)
features = features.reshape(features.shape[0], 1, features.shape[1])
print('Features: {0}\n'.format(features.shape))

# PREDICT
model = load_model('/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/models/model_11.h5')
model.load_weights('/home/uribernal/Desktop/weights_1.3.h5')
predictions = model.predict(features[:, :, :4096], batch_size=1)
print('Predictions: {0}\n'.format(predictions.shape))

# SHOW PREDICTIONS
# calculate root mean squared error
valenceMSE = mean_squared_error(compress_labels(predictions[:, 0, 0])[:-1], compress_labels(y_test[:, 0]))
print('Valence MSE = {0}\n'.format(valenceMSE))
arousalMSE = mean_squared_error(compress_labels(predictions[:, 0, 1])[:-1], compress_labels(y_test[:, 1]))
print('Arousal MSE = {0}\n'.format(arousalMSE))

# calculate PCC
valencePCC = compute_pcc(compress_labels(predictions[:, 0, 0])[:-1], compress_labels(y_test[:, 0]))
print('Valence PCC = {0}\n'.format(valencePCC))
arousalPCC = compute_pcc(compress_labels(predictions[:, 0, 1])[:-1], compress_labels(y_test[:, 1]))
print('Arousal PCC = {0}\n'.format(arousalPCC))

print(predictions, y_test)
