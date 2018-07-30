"""A modification of the mnist_mlp.py example on the Keras github repo.

This file is better suited to run on Cloud ML Engine's servers. It saves the
model for later use in predictions, uses pickled data from a relative data
source to avoid re-downloading the data every time, and handles some common
ML Engine parameters.
"""

from __future__ import print_function

import argparse
import cv2
import h5py  # for saving the model
import io
import keras
import keras.backend as K
import matplotlib.image as mpimg
import numpy as np
import os
from time import time
from datetime import datetime  # for filename conventions
from keras.optimizers import Adam, Adadelta
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Cropping2D, Dropout, Softmax, UpSampling2D, ZeroPadding2D, concatenate
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Reshape
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.lib.io import file_io  # for better file I/O
from generator import create_generators, batch_generator
import models
import sys


#Parameters
INPUT_CHANNELS = 3
NUMBER_OF_CLASSES = 1
IMAGE_W = 640
IMAGE_H = 480

batch_size = 16
num_classes = 1
epochs = 128
patience = 60
target_size = (480, 640) # W, H
steps_per_epoch = 48

loss_name = "binary_crossentropy"

def get_model(batch_size=batch_size):

    inputs = Input((IMAGE_H, IMAGE_W, INPUT_CHANNELS))

    base = models.get_fcn_vgg16_32s(inputs, NUMBER_OF_CLASSES)
    #base = models.get_fcn_vgg16_32s(inputs, NUMBER_OF_CLASSES)
    
    #base = models.get_fcn_vgg16_16s(inputs, NUMBER_OF_CLASSES)
    #base = models.get_fcn_vgg16_8s(inputs, NUMBER_OF_CLASSES)
    #base = models.get_unet(inputs, NUMBER_OF_CLASSES)
    #base = models.get_segnet_vgg16(inputs, NUMBER_OF_CLASSES)

    # sigmoid
    reshape= Reshape((-1,NUMBER_OF_CLASSES))(base)
    act = Activation('sigmoid')(reshape)

    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adadelta(), loss=loss_name)


    #print(model.summary())

    return model

def join_generators(generators):
    while True: # keras requires all generators to be infinite
        data = [next(g) for g in generators]

        x = [d[0] for d in data ]
        y = [d[1] for d in data ]

        yield x, y


# Create a function to allow for different training data and other options
def train_model(image_dir=r'..\data\optical\062A\640x480\images',
                label_dir=r'..\data\optical\062A\640x480\masks',
                job_dir='./tmp/semantic_segmenter',
                **args):
    # set the logging path for ML Engine logging to Storage bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))


    train_generator, validate_generator = create_generators(
        image_dir, label_dir, batch_size=batch_size, target_size=target_size)

    model = get_model()
    model_name = 'model_weights_'+loss_name+'.h5'

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True, verbose=0),
    ]

    history = model.fit_generator(
        train_generator,
        #validation_data=validate_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        #validation_steps=10,
        verbose=1,
        shuffle=False,
        callbacks=callbacks)

    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    # history = model.fit_generator(
    #     train_generator,
    #     validation_data=validate_generator,
    #     epochs=epochs,
    #     steps_per_epoch=48 / batch_size,
    #     validation_steps=8 / batch_size,
    #     verbose=1,
    #     shuffle=False,
    #     callbacks=callbacks)
    #callbacks=[tensorboard])

    # Save the model locally
    model.save('model.h5')

# Create a function to allow for different training data and other options
def train_model_batch_generator(image_dir=None,
                label_dir=None,
                job_dir='./tmp/semantic_segmenter',
                model_out_name="model.h5",
                tissue=True,
                **args):
    # set the logging path for ML Engine logging to Storage bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))

    if image_dir is None:
        if tissue:
            # tissue
            image_dir=r'..\data\optical\062A\640x480\images\images\*.png'
            label_dir=r'..\data\optical\062A\640x480\masks_tissue\masks\*.png'
        else:
            # apperture
            image_dir=r'..\data\optical\062A\640x480\images\images\*.png'
            label_dir=r'..\data\optical\062A\640x480\masks_aperture\masks\*.png'
              
    if tissue:
        model_out_name = "model_tissue.h5"
    else:
        model_out_name = "model_aperture.h5" 

    bg = batch_generator(
        image_dir, 
        label_dir, 
        training_split=0.8)

    model = get_model()
    checkpoint_name = 'model_weights_'+loss_name+'.h5'

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        #ModelCheckpoint(checkpoint_name, monitor='val_loss', save_best_only=True, verbose=0),
    ]

    history = model.fit_generator(
        bg.training_batch(batch_size),
        validation_data=bg.validation_batch(10),
        epochs=epochs,
        steps_per_epoch=bg.steps_per_epoch,
        validation_steps=10,
        verbose=1,
        shuffle=False,
        callbacks=callbacks)

    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    # history = model.fit_generator(
    #     train_generator,
    #     validation_data=validate_generator,
    #     epochs=epochs,
    #     steps_per_epoch=48 / batch_size,
    #     validation_steps=8 / batch_size,
    #     verbose=1,
    #     shuffle=False,
    #     callbacks=callbacks)
    #callbacks=[tensorboard])

    # Save the model locally
    model.save(model_out_name)

def visualy_inspect_result():
    
    model = get_model()
    model.load_weights('model.h5')
    
    bg = batch_generator()
    img,mask= bg.get_random_validation()
    
    y_pred= model.predict(img[None,...].astype(np.float32))[0]
    
    print('y_pred.shape', y_pred.shape)
    
    y_pred= y_pred.reshape((IMAGE_H,IMAGE_W,NUMBER_OF_CLASSES))
    
    print('np.min(y_pred)', np.min(y_pred))
    print('np.max(y_pred)', np.max(y_pred))
    
    cv2.imshow('img',img)
    cv2.imshow('mask 1',mask)
    cv2.imshow('mask object 1',y_pred[:,:,0])
    cv2.waitKey(0)

def make_prediction_movie(image_dir, tissue=True):
    
    model = get_model()

    if tissue:
        model.load_weights('model_tissue.h5')
        vid_name = "tissue.mp4"

        if image_dir:
            bg = batch_generator(image_dir=image_dir,
                                label_dir=None)
        else:
            bg = batch_generator(image_dir=r"..\data\optical\062A\original\raw_image\*.png",
                                label_dir=None)
    else:
        model.load_weights('model_aperture.h5')
        vid_name = "aperture.mp4"
        if image_dir:    
            bg = batch_generator(image_dir=image_dir,
                                label_dir=None)
        else:
            bg = batch_generator(image_dir=r"..\data\optical\062A\original\raw_image\*.png",
                                label_dir=None)        

    vid_fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_out = cv2.VideoWriter(vid_name, vid_fourcc, 15.0, (640,480))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for index in range(bg.image_count):
        img, mask8 = bg.get_image_and_mask(index)
        mask = np.zeros_like(img)
        # mask[:,:,2] = mask8

        y_pred= model.predict(img[None,...].astype(np.float32))[0]
        y_pred= y_pred.reshape((IMAGE_H,IMAGE_W,NUMBER_OF_CLASSES))
        mask[:,:,0] = y_pred[:,:,0] * 255

        alpha = 0.5
        cv2.addWeighted(mask, alpha, img, 1 - alpha, 0, img)

        cv2.putText(img,bg.images[index],(21,21), font, 0.3,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(img,bg.images[index],(20,20), font, 0.3,(255,255,255),1,cv2.LINE_AA)
        vid_out.write(img)
        #cv2.imshow('img',img)
        # key = cv2.waitKey(0)
        # if key == 27: # esc
        #     break
    #cv2.destroyAllWindows()
    vid_out.release()



if __name__ == '__main__':
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--image_dir',
        help='Cloud Storage bucket or local path to image data')
    parser.add_argument(
        '-l', '--label_dir',
        help='Cloud Storage bucket or local path to label data')
    parser.add_argument(
        '-j', '--job_dir',
        default='tmp',
        help='Cloud storage bucket to export the model and store temp files')
    parser.add_argument(
        '-t', '--tissue',
        default=True,
        type=bool,
        help='Process tissue')
    args = parser.parse_args()
    arguments = args.__dict__

    if args.image_dir == None:
        if args.tissue:
            args.image_dir = r'..\data\optical\062A\640x480\images\images\*.png'
            args.label_dir=r"..\data\optical\062A\original\raw_tissue_mask\*.png"
        else:
            args.image_dir = r'..\data\optical\062A\640x480\images\images\*.png'
            args.label_dir=r"..\data\optical\062A\original\raw_tissue_mask\*.png"
   
    #train_model_batch_generator(**arguments)
    make_prediction_movie(args.image_dir, args.tissue)