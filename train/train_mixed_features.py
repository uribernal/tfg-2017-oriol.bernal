import numpy as np
import h5py
from helper import DatasetManager as Dm
from helper import ModelGenerator as Mg
import matplotlib.pyplot as plt
import numpy as np
from helper import DatasetManager as Dm
from helper import ModelGenerator as Mg
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from helper import TelegramBot as Bot


def train_model(model, experiment_id, dropout_probability, batch_size):
    # Path for the data
    data_path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/emotional_impact.h5'

    # Path for the figures
    figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/mixed_features/' + \
                   '{min:05}_Mixed_Features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.png'

    # Path for the weights
    store_weights_file = 'train_with_Mix_features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.hdf5'

    with h5py.File(data_path, 'r') as hdf:
        labels = np.array(hdf.get('labels'))
        video_features = np.array(hdf.get('visual_features'))
        audio_features = np.array(hdf.get('acoustic_features'))
    res = audio_features.shape[0] % labels
    len = audio_features.shape[0] - res
    audio_features = audio_features[:len, :]
    print('Labels shape: {}'.format(labels.shape))
    print('Video features shape: {}'.format(video_features.shape))
    print('Audio features shape: {}'.format(audio_features.shape))


'''
    # Make data fit into model
    audio_features = audio_features.reshape(audio_features.shape[0], audio_features.shape[1] * audio_features.shape[2] *
                                            audio_features.shape[3])
    video_features = video_features.reshape(video_features.shape[0], video_features.shape[1])
    labels = labels.reshape(labels.shape[0], 1, labels.shape[1])
    labels = labels[:, :, :2]

    print('Labels shape: {}'.format(labels.shape))
    print('Video features shape: {}'.format(video_features.shape))
    print('Audio features shape: {}'.format(audio_features.shape))

    data = np.append(video_features, audio_features, axis=1)
    data = data.reshape(video_features.shape[0], 1, video_features.shape[1]+audio_features.shape[1])
    print('Labels shape: {}'.format(labels.shape))
    print('Data shape: {}'.format(data.shape))

    # Make data fit into batches
    res = data.shape[0] % batch_size
    new_len = data.shape[0] - res
    data = data[:new_len]
    labels = labels[:new_len]
    print('Audios shape: {}'.format(data.shape))
    print('Labels shape: {}\n'.format(labels.shape))

    # Make data fit into batches
    res = data.shape[0] % batch_size
    new_len = data.shape[0] - res
    audios = data[:new_len]
    labels = labels[:new_len]
    print('Audios shape: {}'.format(audios.shape))
    print('Labels shape: {}\n'.format(labels.shape))

    # Split data into train and validation
    num = int(0.7 * new_len / batch_size)
    part = num * batch_size  # 70 %
    x_train = audios[0:part, :]
    y_train = labels[0:part]
    x_validation = audios[part:, :]
    y_validation = labels[part:]
    print('Train Input shape: {}'.format(x_train.shape))
    print('Train Output shape: {}\n'.format(y_train.shape))
    print('Validation Input shape: {}'.format(x_validation.shape))
    print('Validation Output shape: {}\n'.format(y_validation.shape))

    # Compiling Model
    print('Compiling model')
    optimizer = Adam(lr=0.0001)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer)
    print('Model Compiled!')

    # Callbacks
    stop_patience = 100
    model_checkpoint = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/' + \
                       store_weights_file.format(experiment_id=experiment_id, batch_size=batch_size,
                                                 drop_out=dropout_probability)

    checkpointer = ModelCheckpoint(filepath=model_checkpoint,
                                   verbose=1,
                                   save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=10,
                                  min_lr=1e-10000,
                                  verbose=1)

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=stop_patience)

    # Training
    train_loss = []
    validation_loss = []

    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,  # Number of samples per gradient update.
                        validation_data=(x_validation, y_validation),
                        verbose=2,
                        nb_epoch=10000,
                        callbacks=[checkpointer, reduce_lr, early_stop],
                        shuffle=False)
    print(history.history.keys())
    train_loss.extend(history.history['loss'])
    validation_loss.extend(history.history['val_loss'])

    min = np.min(validation_loss)

    print('Reseting model states')
    model.reset_states()
    Bot.save_plots(train_loss, validation_loss, figures_path.format(min=min, experiment_id=experiment_id,
                                                                batch_size=batch_size, drop_out=dropout_probability))
    return min


if __name__ == "__main__":
    import time
    drop_out = .5

    for experiment_id in range(1, 30):
        batch_size = 32 * experiment_id
        figures_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/mixed_features/' + \
                       '{min:05}_Mixed_Features_e_{experiment_id}_b{batch_size:03}_d{drop_out:02}.png'

        description = 'Experiment {0}: Mixed Features, Using callbacks, drop-out={1}, batch-size={2}.'.format(
            experiment_id, drop_out, batch_size)

        # Get the LSTM model
        model = Mg.lstm_alberto_tfg_c3d_val_ar(batch_size, 1, drop_out, True)

        Bot.send_message(description)
        start = time.time()
        min = train_model(model, experiment_id, drop_out, batch_size)
        image_path = figures_path.format(min=min, experiment_id=experiment_id, batch_size=batch_size, drop_out=drop_out)

        end = time.time()
        Bot.send_image(image_path)
        Bot.send_elapsed_time(end - start)
    Bot.send_message('Finished')
'''

# Path for the data
data_path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/emotional_impact.h5'
w = np.array([])
movies = Dm.get_movies_names()
del movies[22]
i = 0
for movie in movies:
    #print(movie)
    #movie = 'Wanted'
    with h5py.File(data_path, 'r') as hdf:
        labels = np.array(hdf.get('dev/labels'))
        ground_truth = np.array(hdf.get('dev/ground_truth_data/' + movie))
        video_features = np.array(hdf.get('dev/visual_features/' + movie))
        audio_features = np.array(hdf.get('dev/acoustic_features/' + movie))
    lab = ground_truth.shape[0]
    labels = labels[:,:2]
    i += lab
    res = audio_features.shape[0] % lab
    if(res==0):
        res = lab
        len = audio_features.shape[0] - res
        video_features = video_features[:len]

    len = audio_features.shape[0] - res

    audio_features = audio_features[:len]
    n = video_features.shape[0]
    feat1 = video_features.shape[1]
    feat2 = audio_features.shape[1]
    #print('Labels shape: {}'.format(labels.shape))
    #print('Video features shape: {}'.format(video_features.shape))
    #print('Audio features shape: {}'.format(audio_features.shape))
    data = np.append(video_features, audio_features, axis=1)
    data = data.reshape(n, feat1 + feat2)

    #########################

    data = data.reshape(lab, int(data.shape[0] / lab), data.shape[1])
    #print('Features shape: {}'.format(data.shape))
    w = np.append(w, data)
q = w.reshape(i, 9, 7168)
labels = labels[:4362]
a = labels
for i in range(8):
    labels = np.append(labels, a)
print(labels.shape)
labels = labels.reshape(9, 4362, 2)
labels = labels.transpose(1, 0, 2)

#labels = labels.reshape(labels.shape[0], 1, labels.shape[1])
print('Features shape: {}'.format(q.shape))
print('Labels shape: {}'.format(labels.shape))
model = Mg.lstm_alberto_tfg_c3d_val_ar(1, 9, 0.5, True)
# Compiling Model
print('Compiling model')
optimizer = Adam(lr=0.0001)
model.compile(loss='mean_squared_error', optimizer=optimizer)
print('Model Compiled!')
# Training
train_loss = []
validation_loss = []

history = model.fit(q,
                    labels,
                    batch_size=1,  # Number of samples per gradient update.
                    verbose=2,
                    nb_epoch=10000,
                    shuffle=False)
train_loss.extend(history.history['loss'])
validation_loss.extend(history.history['val_loss'])