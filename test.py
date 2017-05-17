import keras

from old.helper import ModelGenerator as Mg

print(keras.__version__)
model = Mg.C3D_conv_features()