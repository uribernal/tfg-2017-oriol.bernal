from helper import AudioHelper as Ah
from helper import DatasetManager as Dm
from helper import VideoHelper as Vh


# Extract audios
Ah.extract_audios()

# Change FPS in videos to 30
'''
import subprocess
command = 'ffmpeg -r 30 -f image2 -i /home/uribernal/Desktop/MediaEval2016/testset/MEDIAEVAL16-ContinuousPrediction-data/data/MEDIAEVAL16_FM_00.mp4  /home/uribernal/Desktop/MediaEval2016/testset/MEDIAEVAL16-ContinuousPrediction-data/data/caca.mp4'
subprocess.call(command, shell=True)
'''

# Change FS in audios to 44100
