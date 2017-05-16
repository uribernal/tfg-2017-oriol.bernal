from keras.applications.vgg19 import VGG19
from keras.layers import Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

videos_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/' \
              'LIRIS-ACCEDE-continuous-movies/continuous-movies/'
# Path for the weights
store_weights_file = 'Audio_InceptionV3_e_{experiment_id}_e{epoch:03}.hdf5'
experiment_id = 0
epochs = 500
batch_size = 32


# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(98, 64, 1))  # this assumes K.image_data_format() == 'channels_last'

model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False, input_shape=(98, 64, 3))
# Compiling Model
print('Compiling model')
optimizer = Adam(lr=10e-3)
model.compile(loss='mean_squared_error',
              optimizer=optimizer)
print('Model Compiled!')

# Callbacks
stop_patience = 80
model_checkpoint = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/checkpoints/' + \
                   store_weights_file.format(experiment_id=experiment_id, epoch=epochs)

checkpointer = ModelCheckpoint(filepath=model_checkpoint,
                               verbose=1,
                               save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=10,
                              min_lr=0,
                              verbose=1)

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=stop_patience)

# Training
train_loss = []
validation_loss = []

from pipeline.pipeline import get_movies_names, acoustic_data_processed

movies = get_movies_names(videos_path)
acoustic_data = acoustic_data_processed(movies[:5])
print(acoustic_data.shape)
acoustic_data = acoustic_data.reshape(1438*3, 1, 98, 64)
acoustic_data = acoustic_data.transpose(0, 2, 3, 1)
print(acoustic_data.shape)

prediction = model.predict(acoustic_data, batch_size=1)
print(prediction.shape)