import threading
import time

import numpy as np


def import_labels(f):
    ''' Read from a file all the labels from it '''
    lines = f.readlines()
    labels = []
    i = 0
    for l in lines:
        t = l.split('\t')
        assert int(t[0]) == i
        label = t[1].split('\n')[0]
        labels.append(label)
        i += 1
    return labels

def to_categorical(y, nb_classes=None):
    ''' Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def generate_output(video_info, labels, length=16):
    ''' Given the info of the vide, generate a vector of classes corresponding the output for each
    clip of the video which features have been extracted.
    '''
    nb_frames = video_info['num_frames']
    last_first_name = nb_frames - length + 1

    start_frames = range(0, last_first_name, length)

    # Check the output for each frame of the video
    outputs = ['none'] * nb_frames
    for i in range(nb_frames):
        # Pass frame to temporal scale
        t = i / float(nb_frames) * video_info['duration']
        for annotation in video_info['annotations']:
            if t >= annotation['segment'][0] and t <= annotation['segment'][1]:
                outputs[i] = labels.index(annotation['label'])
                label = annotation['label']
                break

    instances = []
    for start_frame in start_frames:
        # Obtain the label for this isntance and then its output
        output = None

        outs = outputs[start_frame:start_frame+length]
        if outs.count(label) >= length / 2:
            output = labels.index(label)
        else:
            output = labels[0]
        instances.append(output)

    return instances

class VideoGenerator(object):

    def __init__(self, videos, stored_videos_path,
            stored_videos_extension, length, input_size):
        self.videos = videos
        self.total_nb_videos = len(videos)
        self.flow_generator = self._flow_index(self.total_nb_videos)
        self.lock = threading.Lock()
        self.stored_videos_path = stored_videos_path
        self.stored_videos_extension = stored_videos_extension
        self.length = length
        self.input_size = input_size

    def _flow_index(self, total_nb_videos):
        pointer = 0
        while pointer < total_nb_videos:
            pointer += 1
            yield pointer-1

    def next(self):
        with self.lock:
            index = next(self.flow_generator)
        t1 = time.time()
        video_id = self.videos[index]
        path = self.stored_videos_path + '/' + video_id + '.' + self.stored_videos_extension
        vid_array = video_to_array(path, start_frame=0,
                                   resize=self.input_size)
        if vid_array is not None:
            vid_array = vid_array.transpose(1, 0, 2, 3)
            nb_frames = vid_array.shape[0]
            nb_instances = nb_frames // self.length
            vid_array = vid_array[:nb_instances*self.length,:,:,:]
            vid_array = vid_array.reshape((nb_instances, self.length, 3,)+(self.input_size))
            vid_array = vid_array.transpose(0, 2, 1, 3, 4)
        t2 = time.time()
        print('Time to fetch {} video: {:.2f} seconds'.format(video_id, t2-t1))
        return video_id, vid_array

    def __next__(self):
        self.next()

def video_to_array(video_path, resize=None, start_frame=0, end_frame=None,
                   length=None, dim_ordering='th'):
    ''' Convert the video at the path given in to an array
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
    Returns:
        video (nparray): Array with all the data corresponding to the video
                         given. Order of dimensions are: channels, length
                         (temporal), height, width.
    Raises:
        Exception: If the video could not be opened
    '''

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
    ''' Return the number of frames of the video track of the video given '''
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
    ''' Return the duration of the video track of the video given '''
    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        CAP_PROP_FPS = cv2.CAP_PROP_FPS
    else:
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        CAP_PROP_FPS = cv2.cv.CV_CAP_PROP_FPS

    cap = cv2.CaptureFromFile(video_path)
    if not cap.isOpened():
        raise Exception('Could not open the video')
    num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(CAP_PROP_FPS))
    duration = num_frames / fps
    # When everything done, release the capture
    cap.release()
    return duration