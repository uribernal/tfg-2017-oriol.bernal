from helper import ModelGenerator as Mg
import keras
print(keras.__version__)
model = Mg.C3D_conv_features()