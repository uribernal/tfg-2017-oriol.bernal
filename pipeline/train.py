import numpy as np
import matplotlib.pyplot as plt
import h5py


def save_plots(iteration, train_loss, validation_loss, experiment_id):
    path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/'
    file = 'Train2017_emotion_classification_{experiment_id}_e{epoch:03}.png'

    # Show plots
    x = np.arange(len(validation_loss))
    fig = plt.figure(1)
    fig.suptitle('LOSS', fontsize=14, fontweight='bold')

    # LOSS: TRAINING vs VALIDATION
    plt.plot(x, train_loss, '--', linewidth=2, label='train')
    plt.plot(x, validation_loss, label='validation')
    plt.legend(loc='upper right')

    # MIN
    val, idx = min((val, idx) for (idx, val) in enumerate(validation_loss))
    plt.annotate(str(val), xy=(idx, val), xytext=(idx, val - 0.01),
                 arrowprops=dict(facecolor='black', shrink=0.0005))

    plt.savefig(path + file.format(experiment_id=experiment_id, epoch=iteration), dpi=fig.dpi)
    plt.close()


def get_movies_names(path: str):
    from os import listdir
    from os.path import isfile, join

    videos = [f for f in listdir(path) if isfile(join(path, f))]
    for cont, file in enumerate(videos):
        videos[cont] = file[:-4]

    return videos


def get_movie_labels(path: str, movie: str):
    extensions = ['-MEDIAEVAL2017-valence_arousal.txt', '-MEDIAEVAL2017-fear.txt']
    time_1 = []
    mean_arousal = []
    mean_valence = []
    time_2 = []
    fear = []
    for cte, ext in enumerate(extensions):
        with open(path + movie + ext) as f:
            for line in f:
                if cte == 0:
                    l = line.strip().split()
                    time_1.append(l[1])
                    mean_valence.append(l[2])
                    mean_arousal.append(l[3])
                elif cte == 1:
                    l = line.strip().split()
                    time_2.append(l[1])
                    fear.append(l[2])
                else:
                    raise Exception('Error extracting labels')
        if cte == 0:
            time_1.pop(0)
            mean_valence.pop(0)
            mean_arousal.pop(0)
        elif cte == 1:
            time_2.pop(0)
            fear.pop(0)
        else:
            raise Exception('Error extracting labels')

    lab = np.array((mean_valence, mean_arousal, fear), dtype=float)
    return lab


def get_ground_truth_data(videos: list, show_info: str = False):
    lab = np.array([])
    for video in videos:
        l = get_movie_labels(annotations_path, video)  # The ground truth data for each film
        e = l.transpose(1, 0)
        lab = np.append(lab, e)
        if show_info:
            print('{0}: {1}'.format(video, l.shape))
    if show_info:
        print('')
    return lab.reshape(lab.shape[0] // 3, 3)


def get_predictions_length(videos: list, show_info: str=False):
    res = []
    for video in videos:
        l = get_movie_labels(annotations_path, video)  # The ground truth data for each film
        e = l.transpose(1, 0)
        res.append(e.shape[0])

    return res


def get_videos_info(videos: list, ext: str):
    from helper import VideoHelper as Vh
    rgb_frames = []
    video_fps = []
    video_duration = []
    for cont, movie in enumerate(videos):
        rgb_frames.append(Vh.get_num_frames(
            videos_path + movie + ext))  # The number of frames per video (depend on its length and fps)
        video_fps.append(Vh.get_fps(videos_path + movie + ext))  # The number of fps per video
        video_duration.append(Vh.get_duration(videos_path + movie + ext))

    return rgb_frames, video_fps, video_duration


def read_video(input_video: str, resize: tuple=(112, 112)):
    from helper import VideoHelper as Vh
    vid = Vh.video_to_array(input_video, resize=resize)

    return vid


def get_resized_video(index_video, video_path, input_size):
    video_array = read_video(video_path, input_size)
    print('{0}: {1}'.format(movies[index_video], video_array.shape))
    lab = predictions_length[index_video]
    resized_number_of_frames = lab * 5 * int(np.round(fps[index_video]))
    video_array = video_array[:, :resized_number_of_frames, :, :]
    video_array = video_array.transpose(1, 0, 2, 3)
    video_array = video_array.reshape((lab, int(video_array.shape[0]/lab), 3, input_size[0], input_size[1]))
    print('resized {0}: {1}'.format(movies[index_video], video_array.shape))
    return video_array


def save_visual_data(videos_path: str, videos: list, videos_extension: str, file: str):
    for cont, video in enumerate(videos):
        input_video = videos_path + video + videos_extension

        visual = get_resized_video(cont, input_video, (98, 64))
        with h5py.File(file, 'r+') as hdf:
            # Store data
            hdf.create_dataset('features/'+video, data=visual, compression='gzip', compression_opts=9)
        print(video + ' Stored:')


def get_visual_data(database_path: str, videos: list, predictions_length: list):
    data = np.array([])
    cont = 0
    for i, video in enumerate(videos):
        with h5py.File(database_path, 'r') as hdf:
            # Read movie
            mov = hdf.get('features/' + video)
            data = np.append(data, mov)
        cont += predictions_length[i]
    return data.reshape(cont, int(data.shape[0]/(cont*3*98*64)), 3, 98, 64)


file = '/home/uribernal/Desktop/MediaEval2017/data/visual_data.h5'
# Create the HDF5 file
#hdf = h5py.File(file, 'w')
#hdf.close()


annotations_path = '/home/uribernal/Desktop/MediaEval2017/annotations/'

videos_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/' \
              'LIRIS-ACCEDE-continuous-movies/continuous-movies/'

videos_extension = '.mp4'

movies = get_movies_names(videos_path)  # Names of the videos from the DB
labels = get_ground_truth_data(movies)  # The ground truth data for each film
predictions_length = get_predictions_length(movies)  # The number of predictions per video (depend on its length)
frames, fps, duration = get_videos_info(movies, videos_extension)
#save_visual_data(videos_path, movies, videos_extension, file)
visual_data = get_visual_data(file, movies, predictions_length)
print('visual_data {0}: {1}'.format(visual_data.shape, visual_data))


