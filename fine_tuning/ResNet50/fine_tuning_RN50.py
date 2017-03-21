#https://github.com/imatge-upc/retrieval-2016-remote/blob/64c300a00d3671105f930ad8d927122f6048d04e/fine_tuning_ResNet50.py

from keras.applications.resnet50 import ResNet50
from keras.models import Model
import numpy as np
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from mlxtend.preprocessing import one_hot
import matplotlib

# create the base pre-trained model
base_model = ResNet50(weights='imagenet')#, include_top=False)

###     base_model Architecture
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

img_path = '/home/uribernal/Desktop/code/test/'

# add a global spatial average pooling layer
x = base_model.output
print(x.shape)
print(x)

# let's add a fully-connected layer
fc1 = Dense(1000, activation='tanh')(x)
b2 = BatchNormalization(axis=1)(fc1)
d1 = Dropout(0.6)(b2)
fc2 = Dense(1000, activation='tanh', name='fc2')(d1)
b3 = BatchNormalization(axis=1)(fc2)
d2 = Dropout(0.6)(b3)
predictions = Dense(1, activation='sigmoid')(d2)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional ResNet50 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='binary_crossentropy')

# IMAGES LOAD
from continuousMovies import readVideos
videos = readVideos.getClips(100)
print('-----------------videos.shape:')
print(videos.shape)

trainFeatures = np.array([])
# LABELS LOAD
ids, names, labels = readVideos.getLabels()
a = labels[:,2]
r = a[:100]
print('-----------------lab.shape')
print(r.shape)
#r = np.array([])


# train the model on the new data for a few epochs
#model.fit_generator(...)
model.fit(trainFeatures, r, batch_size=32, nb_epoch=10, verbose=2)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:162]:
   layer.trainable = False
for layer in model.layers[162:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0000001, momentum=0.9), loss='binary_crossentropy')
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#model.fit_generator(...)
model.fit(trainFeatures,r,batch_size=32,validation_split=0.33,nb_epoch=10,verbose=2)

print("saving model and weights")
model_json = model.to_json()

with open("ResNet50.json", "w") as json_file:
	json_file.write(model_json)
print("saving...")
model.save_weights('resnet50_weights.h5')