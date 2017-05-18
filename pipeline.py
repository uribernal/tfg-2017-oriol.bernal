
from helper import AudioHelper as Ah
from helper import DatasetManager as Dm
from helper import VideoHelper as Vh


# PRE-WORK
# Extract audios
# Change fps in videos

# CONSTANTS
time_prediction_constant = 10
time_shift_prediction_constant = 5
input_size = (112, 112)
num_frames = 16
num_visual_feat = 4096  # Features after C3D convolutions (visual feature extractor model)
num_acoustic_feat = 4096  # Features after CD convolutions (acoustic feature extractor model)
#win_frames = int(960 * 44100 * 1e-3)  # Length of window frames (no overlapping)
win_frames = 44100
num_stft = 98
num_filter_banks = 64

# THE TASK
print('\033[94m' + 'TASK INFORMATION:' + '\033[0m')
movies = Dm.get_movies_names()  # Names of the videos from the DB
print('films ({0}): {1}'.format(len(movies), movies))
labels_type = ['valence', 'arousal', 'fear']  # Types of ground truth data per video
print('types of labels ({0}): {1}\n'.format(len(labels_type), labels_type))

# Calculate
print('\033[94m' + 'GROUND TRUTH INFORMATION:' + '\033[0m')
labels = Dm.get_ground_truth_data(movies)  # The ground truth data for each film
print('labels: {0}\n'.format(labels.shape))

print('\033[94m' + 'VISUAL INFORMATION:' + '\033[0m')
frames, fps, duration = Dm.get_videos_info(movies)
print('frames ({0}): {1}'.format(len(frames), frames))
print('fps ({0}): {1}'.format(len(fps), fps))
print('duration ({0}): {1}\n'.format(len(duration), duration))

print('\033[94m' + 'ACOUSTIC INFORMATION:' + '\033[0m')
fs, num_audio_samples = Dm.get_audios_info(movies)
print('fs ({0}): {1}'.format(len(fs), fs))
print('num_audio_samples ({0}): {1}\n'.format(len(num_audio_samples), num_audio_samples))

print('\033[94m' + 'PREDICTION INFORMATION:' + '\033[0m')
predictions_length = Dm.get_predictions_length(movies)  # The number of predictions per video (depend on its length)
print('predictions_length ({0}): {1}\n'.format(len(predictions_length), predictions_length))

print('\033[94m' + 'RGB DATA:' + '\033[0m')
visual_data = Vh.get_visual_data(movies[:1], input_size=input_size,
                                 num_frames=num_frames, print_info=True)  # Matrix with the RGB info from videos
print('visual_data ({0})\n'.format(visual_data.shape))

print('\033[94m' + 'AUDIO DATA:' + '\033[0m')
acoustic_data = Ah.get_resized_audios(movies[:2], win_frames=win_frames, print_info=True)  # Matrix with the audio samples from videos
print('acoustic_data {0}\n'.format(acoustic_data.shape))

print('\033[94m' + 'MIXED DATA:' + '\033[0m')
#data = Dm.get_mixed_data(movies[:2], win_frames, print_info=True)
