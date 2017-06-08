# https://github.com/fchollet/deep-learning-models
import numpy as np
from fine_tuning.Audio.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

model = 2

if model == 0:
    from old.fine_tuning.Audio.resnet50 import ResNet50
    model = ResNet50(weights='imagenet')
elif model == 1:
    from old.fine_tuning.Audio.vgg16 import VGG16
    model = VGG16(weights='imagenet', include_top=False)
elif model == 2:
    from old.fine_tuning.Audio.vgg19 import VGG19
    from keras.models import Model
    base_model = VGG19(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)


img_path = '/home/uribernal/Pictures/elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
print('img_to_array Shape: {}'.format(x.shape))
x = np.expand_dims(x, axis=0)
print('expand Shape: {}'.format(x.shape))
x = preprocess_input(x)
print('preprocess_Input Shape: {}'.format(x.shape))

preds = model.predict(x)
print(preds.shape)
print('Predicted:', decode_predictions(preds))