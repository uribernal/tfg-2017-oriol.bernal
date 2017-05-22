import time
from helper import TelegramBot as Bot
from helper import ModelGenerator as Mg
from train.train_mixed_features import train_model as T1
from train.train_audio_features import train_model as T2
from train.train_raw_audio import train_model as T3

drop_out = .5

for experiment_id in range(30, 60):
    batch_size = 32 * experiment_id
    figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/mixed_features/' + \
                   '{min:05}_Mixed_Features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.png'

    description = 'Experiment {0}: Mixed Features, Using callbacks, drop-out={1}, batch-size={2}.'.format(
        experiment_id, drop_out, batch_size)

    # Get the LSTM model
    model = Mg.lstm_alberto_tfg_c3d_val_ar(batch_size, 1, drop_out, True)

    Bot.send_message(description)
    start = time.time()
    min = T1(model, experiment_id, drop_out, batch_size)
    image_path = figures_path.format(min=min, experiment_id=experiment_id, batch_size=batch_size, drop_out=drop_out)

    end = time.time()
    Bot.send_image(image_path)
    Bot.send_elapsed_time(end - start)
Bot.send_message('Finished')

for experiment_id in range(30, 60):
    batch_size = 32 * experiment_id
    figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/audio_features/' + \
                   '{min:05}_Audio_Features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.png'
    description = 'Experiment {0}: Audio_Features, Using callbacks, drop-out={1}, batch-size={2}.'.format(
        experiment_id, drop_out, batch_size)

    # Get the LSTM model
    model = Mg.lstm_audio_features(batch_size, True)

    Bot.send_message(description)
    start = time.time()
    min = T2(model, experiment_id, drop_out, batch_size)
    image_path = figures_path.format(min=min, experiment_id=experiment_id, batch_size=batch_size, drop_out=drop_out)

    end = time.time()
    Bot.send_image(image_path)
    Bot.send_elapsed_time(end - start)
Bot.send_message('Finished')

for experiment_id in range(30, 60):
    batch_size = 32 * experiment_id
    figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/raw_audio/' + \
                   '{min:05}_Raw_Audio_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.png'

    description = 'Experiment {0}: Raw Audio, Using callbacks, drop-out={1}, batch-size={2}.'.format(
        experiment_id, drop_out, batch_size)

    # Get the LSTM model
    model = Mg.lstm_raw_audio(batch_size, True)

    Bot.send_message(description)
    start = time.time()
    min = T3(model, experiment_id, drop_out, batch_size)
    image_path = figures_path.format(min=min, experiment_id=experiment_id, batch_size=batch_size, drop_out=drop_out)
    end = time.time()
    Bot.send_image(image_path)
    Bot.send_elapsed_time(end - start)
Bot.send_message('Finished')