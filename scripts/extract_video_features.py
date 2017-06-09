"""
This script allows to extract all the audios from the movies_path
or it extracts a single audio from movie_path into the audio_path
using the function extract_audio
"""

from helper import DatasetManager as Dm
from fine_tuning.Video.extract_c3d_features import extract_features


def extract_video_features(video_name, video_path):
    features = extract_features(video_name, video_path=video_path, saved_data=False)
    return features


if __name__ == '__main__':
    from helper.DatasetManager import features_path

    movies = Dm.get_movies_names()
    for movie in movies:
        _ = extract_features(movie, features_path=features_path, saved_data=True)
