from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt



#Load Pre-trained model from model zoo
#We load the pre-trained VGG16 model from the Keras model zoo and list out the various layers in it.
#First time you invoke the VGG16 constructor, it will download the trained model to ~/.keras/models.
#Looks like there is 5 blocks of convolution+pooling followed by 2 fully connected layers.

vgg16_model = VGG16(weights='imagenet', include_top=True)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#vgg16_model.compile(optimizer=sgd, loss='categorical_crossentropy')
vgg16_model.summary()




#Visualizing Inputs and Outputs
#We take an input image and look at what it turns into after the first 2 layers of convolution and pooling.
lena = image.load_img("/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/fine_tuning/lena.png", target_size=(224, 224))
plt.imshow(lena)
lena = image.img_to_array(lena)
lena = np.expand_dims(lena, axis=0)
lena = preprocess_input(lena)
print(lena.shape) #ndayyay of shape (1, 224, 224, 3)


model = Model(input=vgg16_model.input,
              output=vgg16_model.get_layer('block1_pool').output)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

block1_pool_features = model.predict(lena)
print(block1_pool_features.shape) #(1,112,112,64)
print(block1_pool_features) #(1,112,112,64)

'''


fig, axes = plt.subplots(8, 8, figsize=(10, 10))
axes = np.ravel(axes)
for i in range(block1_pool_features.shape[3]):
    axes[i].imshow(255-block1_pool_features[0, :, :, i], interpolation="nearest")
    axes[i].set_xticks([])
    axes[i].set_yticks([])
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

'''