"""
This script fine tunes VGG19 to extract features from audio inputs
"""

from helper import DatasetManager as Dm
from fine_tuning.Audio.extract_vgg19_features import extract_features


def extract_audio_features(audio_name, audio_path):
    features = extract_features(audio_name, audio_path=audio_path, saved_data=False)
    return features


if __name__ == '__main__':

    from helper.DatasetManager import features_path

    movies = Dm.get_movies_names()
    for movie in movies:
        _ = extract_features(movie, features_path=features_path, saved_data=True)
