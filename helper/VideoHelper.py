"""
This assistant allows to compute different properties 
from videos such as duration and frames.
"""

import cv2
import numpy as np


def video_to_array(video_path, resize=None, start_frame=0, end_frame=None,
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
        video = video.transpose(3, 0, 1, 2)
    return video


def get_num_frames(video_path):
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


def get_duration(video_path):
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


def get_fps(video_path):
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


def reproduce_video(video_path):
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


def read_video(input_video: str, resize: tuple=(112, 112)):
    vid = video_to_array(input_video, resize=resize)

    return vid


def get_visual_data(videos: list, input_size, num_frames, print_info=False):
    from helper import DatasetManager as Dm
    data = np.array([])
    for cont, video in enumerate(videos):
        input_video = Dm.get_video_path(video)
        video_array = read_video(input_video, input_size)
        if print_info:
            print('{0} shape: {1}'.format(video, video_array.shape))
        nb_frames = get_num_frames(input_video)
        nb_clips = nb_frames // num_frames
        video_array = video_array.transpose(1, 0, 2, 3)
        video_array = video_array[:nb_clips * num_frames, :, :, :]
        video_array = video_array.reshape((nb_clips, num_frames, 3, 112, 112))
        video_array = video_array.transpose(0, 2, 1, 3, 4)
        if print_info:
            print('resized {0}: {1}\n'.format(video, video_array.shape))
        data = np.append(data, video_array)

    return data.reshape(int(data.shape[0]/(3*16*112*112)), 3, 16, 112, 112)


def get_resized_video(index_video, video_path, input_size, print_info=False):
    from helper import DatasetManager as Dm
    movies = Dm.get_movies_names()
    predictions_length = Dm.get_predictions_length(movies)
    fps = get_fps(video_path)
    video_array = read_video(video_path, input_size)
    if print_info:
     print('{0} shape: {1}'.format(movies[index_video], video_array.shape))
    lab = predictions_length[index_video]
    resized_number_of_frames = lab * 5 * int(np.round(fps))
    video_array = video_array[:, :resized_number_of_frames, :, :]
    video_array = video_array.transpose(1, 0, 2, 3)
    video_array = video_array.reshape((lab, int(video_array.shape[0]/lab), 3, input_size[0], input_size[1]))
    if print_info:
        print('resized {0}: {1}'.format(movies[index_video], video_array.shape))
    return video_array


def get_visual(videos: list, print_info=False):
    from helper.DatasetManager import videos_path, videos_extension
    visual = np.array([])
    for cont, video in enumerate(videos):
        input_video = videos_path + video + videos_extension

        resized_video = get_resized_video(cont, input_video, (98, 64), print_info=print_info)
        resized_video = resized_video[:, :120, :, :, :]
        visual = np.append(visual, resized_video)
    return visual.reshape(int(visual.shape[0]/(120*3*98*64)), 120, 3, 98, 64)


