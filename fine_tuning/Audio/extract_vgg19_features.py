import h5py
import os.path
from keras.applications.vgg19 import VGG19
from helper import AudioHelper as Ah
from keras.layers import Input
import numpy as np


def get_vgg19_model():
    print('Loading VGG19 network...')
    # this could also be the output a different Keras model or layer
    input_tensor = Input(shape=(3, 96, 64))
    model = VGG19(input_tensor=input_tensor, weights=None, include_top=False, input_shape=None)
    model.load_weights('/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/fine_tuning/Audio/vgg19_weights.h5')
    return model


def extract_features(audio_name, audio_path=None, audio_extension=None, features_path=None, saved_data=False,
                     win_frames=23520, num_stft=512, num_filter_banks=64):
    if features_path is not None:
        # If file doesn't exist, create it
        if not os.path.isfile(features_path):
            # Create the HDF5 file
            hdf = h5py.File(features_path, 'w')
            hdf.close()

    model = get_vgg19_model()
    if saved_data:
        from helper.DatasetManager import data_path
        with h5py.File(data_path, 'r') as hdf:
            acoustic_data = np.array(hdf.get('dev/acoustic_data/' + audio_name))
    else:
        if audio_path is None:
            from helper.DatasetManager import audios_path as audio_path
        if audio_extension is None:
            from helper.DatasetManager import audios_extension as audio_extension

        input_audio = audio_path + audio_name + audio_extension
        resized_audio = Ah.get_resized_audio(input_audio, win_frames=win_frames, print_info=False)
        acoustic_data = Ah.compute_stft_and_melbank(resized_audio, num_stft, num_filter_banks, print_info=False)
        print(acoustic_data.shape)

    cte = acoustic_data.shape[0]
    a = np.append(acoustic_data, acoustic_data)
    acoustic_data = np.append(a, acoustic_data)
    computed_audio = acoustic_data.reshape(3, cte, 96, 64)
    #  print(computed_audio.shape)
    computed_audio = computed_audio.transpose(1, 0, 2, 3)
    #  print(computed_audio.shape)

    print('Extracting features...')
    predictions = model.predict(computed_audio[:, :, :, :], batch_size=1)
    predictions = predictions.reshape(predictions.shape[0],
                                      predictions.shape[1] * predictions.shape[2] * predictions.shape[3])
    #  print(predictions.shape)

    # If h5py file path is given, save features in the h5py file
    if features_path is not None:
        with h5py.File(features_path, 'r+') as hdf:
            hdf.create_dataset('dev/acoustic_features/' + audio_name, data=predictions,
                               compression='gzip', compression_opts=9)
        print('{} stored'.format(audio_name))

    return predictions
