"""
This assistant allows to compute different properties 
from videos such as duration and frames.
"""

import cv2
import numpy as np
import subprocess
from helper import DatasetManager as Dm


def video_2_30fps(video_input_path: str, video_output_path: str, video_shape: tuple =None):
    """ Convert a video to 30 fps and store it in video_output_path.
    If video_shape is indicated it also changes video's width and height. """

    if video_shape is None:
        s = ''
    else:
        s = '-s ' + str(video_shape[0]) + 'x' + str(video_shape[1])

    command = 'ffmpeg -y -i ' + video_input_path + ' -r 30 ' + s + \
              ' -c:v libx264 -b:v 3M -strict -2 -movflags faststart '+video_output_path
    subprocess.call(command, shell=True)


def videos_2_30fps(videos_path=None, resized_videos_path=None, videos_extension=None, video_shape=None):
    """ Convert all the videos to 30 fps and store it in resized_videos_path.
    If video_shape is indicated it also changes videos' width and height. """

    if videos_path is None:
        from helper.DatasetManager import videos_path
    if resized_videos_path is None:
        from helper.DatasetManager import resized_videos_path
    if videos_extension is None:
        from helper.DatasetManager import videos_extension

    movies = Dm.get_movies_names()
    for movie in movies:
        input_path = videos_path + movie + videos_extension
        output_path = resized_videos_path + movie + videos_extension

        video_2_30fps(input_path, output_path, video_shape=video_shape)


def video_to_array(video_path: str, resize=None, start_frame=0, end_frame=None,
                   length=None, dim_ordering='th'):
    """ Convert the video at the path given in to an array
    Args:
        video_path (string): path where the video is stored
        resize (Optional[tupple(int)]): desired size for the output video.
            Dimensions are: height, width
        start_frame (Optional[int]): Number of the frame to start to read
            the video
        end_frame (Optional[int]): Number of the frame to end reading the
            video.
        length (Optional[int]): Number of frames of length you want to read
            the video from the start_frame. This override the end_frame
            given before.
        dim_ordering (Optional[str]): ...
    Returns:
        video (nparray): Array with all the data corresponding to the video
                         given. Order of dimensions are: channels, length
                         (temporal), height, width.
    Raises:
        Exception: If the video could not be opened
    """

    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
    else:
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        CAP_PROP_POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES

    if dim_ordering not in ('th', 'tf'):
        raise Exception('Invalid dim_ordering')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception('Could not open the video')

    num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
    if start_frame >= num_frames or start_frame < 0:
        raise Exception('Invalid initial frame given')
    # Set up the initial frame to start reading
    cap.set(CAP_PROP_POS_FRAMES, start_frame)
    # Set up until which frame to read
    if end_frame:
        end_frame = end_frame if end_frame < num_frames else num_frames
    elif length:
        end_frame = start_frame + length
        end_frame = end_frame if end_frame < num_frames else num_frames
    else:
        end_frame = num_frames
    if end_frame < start_frame:
        raise Exception('Invalid ending position')

    frames = []
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            return None

        if resize:
            # The resize of CV2 requires pass firts width and then height
            frame = cv2.resize(frame, (resize[1], resize[0]))
        frames.append(frame)

    video = np.array(frames, dtype=np.float32)
    if dim_ordering == 'th':
        video = np.transpose(video, (3, 0, 1, 2))
    return video


def get_num_frames(video_path: str):
    """ Return the number of frames of the video track of the video given """

    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT

    else:
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception('Could not open the video')
    num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
    return num_frames


def get_duration(video_path: str):
    """ Return the duration of the video track of the video given """

    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        CAP_PROP_FPS = cv2.CAP_PROP_FPS

    else:
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        CAP_PROP_FPS = cv2.cv.CV_CAP_PROP_FPS

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception('Could not open the video')
    num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(CAP_PROP_FPS))
    duration = num_frames / fps
    # When everything done, release the capture
    cap.release()
    return duration


def get_fps(video_path: str):
    """ Return the fps of the video given """

    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FPS = cv2.CAP_PROP_FPS

    else:
        CAP_PROP_FPS = cv2.cv.CV_CAP_PROP_FPS

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception('Could not open the video')
    fps = float(cap.get(CAP_PROP_FPS))
    # When everything done, release the capture
    cap.release()
    return fps


def reproduce_video(video_path: str):
    """ Reproduces the video in RGB """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.open(video_path)

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def get_resized_video(video: str, video_path=None, video_extension=None, input_size=(112, 112), print_info=False):
    """ Returns the video in a matrix form (num_clips, 3, 16, width, height).
    Clips are sets of 16 frames and the 3 is because of the RGB channels """

    if video_path is None:
        from helper.DatasetManager import videos_path as video_path
    if video_extension is None:
        from helper.DatasetManager import videos_extension as video_extension

    input_video = video_path + video + video_extension
    video_array = video_to_array(input_video, resize=input_size)
    if print_info:
        print('Video shape: {0}'.format(video_array.shape))
    video_array = video_array.transpose(1, 0, 2, 3)
    chop = video_array.shape[0] // 16
    video_array = video_array[:chop * 16, :, :, :]
    video_array = video_array.reshape(video_array.shape[0] // 16, 16, video_array.shape[1], video_array.shape[2],
                                      video_array.shape[3])
    video_array = video_array.transpose(0, 2, 1, 3, 4)
    if print_info:
        print('Resized video: {0}\n'.format(video_array.shape))
    return video_array
