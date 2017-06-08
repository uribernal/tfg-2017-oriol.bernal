"""
This script allows to convert all the videos from the movies_path
to 30 fps or it converts a single video from movie_path into the 30fps_path
using the function video_2_30fps
"""

from helper import VideoHelper as Vh


def video_2_30fps(input_video_path, output_video_path, video_shape=None):
    Vh.video_2_30fps(input_video_path, output_video_path, video_shape=video_shape)


if __name__ == '__main__':
    video_shape = (112,112)
    Vh.videos_2_30fps(video_shape=video_shape)