
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys

print (os.getcwd())

from keras.applications.densenet import DenseNet121
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.vgg16 import VGG16
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Cropping2D, Dropout, Softmax, UpSampling2D, ZeroPadding2D, concatenate
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model, Sequential


def denseNet(target_size=(480, 640)):
    model = DenseNet121(
        include_top=False,
        weights=None,
        input_shape=(target_size[0], target_size[1], 3),
        pooling=max
        )
    return model

# model = denseNet()

# def mobilenetv2()
# model = MobileNetV2(
#     input_shape=None,
#     alpha=1.0,
#     depth_multiplier=1,
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     pooling=None,
#     classes=1000)




def vgg16(dropout=0.5, target_size=(600, 800)):
    vgg16 = VGG16(include_top=False, input_shape=(target_size[0], target_size[1], 3))
    for layer in vgg16.layers:
        layer.trainable = False

    block3_pool = vgg16.get_layer('block3_pool')
    block4_pool = vgg16.get_layer('block4_pool')
    block5_pool = vgg16.get_layer('block5_pool')

    x = Conv2D(filters=4096, kernel_size=7, padding="same", activation="relu", name="fc6")(block5_pool.output)
    x = BatchNormalization()(x)
    #x = Dropout(dropout, name="fc6_dropout")(x)
    x = Conv2D(filters=4096, kernel_size=1, padding="same", activation="relu", name="fc7")(x)
    x = BatchNormalization()(x)
    #x = Dropout(dropout, name="fc7_dropout")(x)
    x = Conv2D(filters=3, kernel_size=1, padding="same", activation="relu", name="fcn_32")(x)
    #x = UpSampling2D(size=2, name="fcn32_2x")(x)
    x = Conv2DTranspose(filters=3, kernel_size=4, strides=(2, 2), padding="same", name="fcn32_2x")(x)
    #x = ZeroPadding2D(padding=(1, 0), name="fcn32_2x_pad")(x)
    #x = Cropping2D(cropping=((0, 1), (0, 0)), name="fcn32_2x_crop")(x)

    block4_pred = Conv2D(filters=3, kernel_size=1, padding="same", activation="relu", name="block_4_pred")(block4_pool.output)
    x = concatenate([x, block4_pred], name="fcn_16")
    
    #x = UpSampling2D(size=2, name="fcn16_2x")(x)
    x = Conv2DTranspose(filters=3, kernel_size=4, strides=(2, 2), padding="same", name="fcn16_2x")(x)
    #x = ZeroPadding2D(padding=(0, 1), name="fcn16_2x_pad")(x)
    #x = Cropping2D(cropping=((0, 0), (0, 1)), name="fcn16_2x_crop")(x)
    block3_pred = Conv2D(filters=3, kernel_size=1, padding="same", activation="relu", name="block_3_pred")(block3_pool.output)
    x = concatenate([x, block3_pred], name="fcn_8")

    x = Conv2D(filters=3, kernel_size=1, padding="same", activation="relu", name="prediction")(x)
    #x = UpSampling2D(size=8, name="pred_8x")(x)
    x = Conv2DTranspose(filters=3, kernel_size=16, strides=(8, 8), padding="same", name="pred_8x")(x)
    x = ZeroPadding2D(padding=(2, 2), name="pred_8x_pad")(x)
    x = Cropping2D(cropping=((0, 1), (0, 1)), name="fcn8_2x_crop")(x)
    x = Softmax(name="pred_softmax")(x)

    return Model(vgg16.input, x, name="sem_seg")



def get_fcn_vgg16_32s(inputs, n_classes):
    
    # vgg16 = VGG16(input_tensor = inputs, weights=None, include_top=False, pooling='avg')
    # return Model(vgg16.input, vgg16, name="sem_seg")


    x = BatchNormalization()(inputs)
    
    # Block 1
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = Dropout(0.35)(x)

    # Block 4
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = Dropout(0.35)(x)

    # Block 5
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = Dropout(0.35)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding="same")(x)

    #     
    x = Conv2DTranspose(n_classes, kernel_size=(64, 64), strides=(32, 32), activation='linear', padding='same')(x)
    
    return x

def get_segnet_vgg16(inputs, n_classes):
    
    x = BatchNormalization()(inputs)
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Up Block 1
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
    # Up Block 2
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
    # Up Block 3
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    # Up Block 4
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Up Block 5
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    x = Conv2D(n_classes, (1, 1), activation='linear', padding='same')(x)
    
    return x
