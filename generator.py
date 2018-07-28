import glob
import numpy as np
import cv2
from random import randint

def create_generators(image_dir=r'..\data\optical\062A\640x480\images',
                      label_dir=r'..\data\optical\062A\640x480\masks',
                      batch_size=1,
                      target_size=(480, 640)):
    ''' returns a tuple of training and validation ImageDataGenerators '''
    from keras.preprocessing.image import ImageDataGenerator

    data_gen_args = dict(
        validation_split=0.15,
        rescale=1.0 / 255,
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        #rotation_range=90.,
        #width_shift_range=0.1,
        #height_shift_range=0.1,
        #zoom_range=0.2
    )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    # image_datagen.fit(image_dir, augment=True, seed=seed)
    # mask_datagen.fit(label_dir, augment=True, seed=seed)

    flow_args = dict(
        batch_size=batch_size,
        target_size=target_size,
        # class_mode=None,
        seed=seed)

    image_generator = image_datagen.flow_from_directory(
        image_dir, subset='training', **flow_args)

    mask_generator = mask_datagen.flow_from_directory(
        label_dir, subset='training', color_mode='grayscale', class_mode='binary', **flow_args)

    validate_image_generator = image_datagen.flow_from_directory(
        image_dir, subset='validation', **flow_args)

    validate_mask_generator = mask_datagen.flow_from_directory(
        label_dir, subset='validation', **flow_args)

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    validate_generator = zip(validate_image_generator, validate_mask_generator)
    return (train_generator, validate_generator)


class batch_generator(object):
    ''' generator for images and masks '''
    def __init__(self,
             image_dir=r'..\data\optical\062A\640x480\images\images\*.png', 
             label_dir=None, 
             training_split=0.8):
        ''' foo '''
        self.training_split = training_split
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = glob.glob(image_dir)
        self.image_count = len(self.images)
        self.masks = None
        if self.label_dir:
            self.masks = glob.glob(label_dir)
            self.mask_count = len(self.masks)
            assert (self.image_count == self.mask_count)
        training_set = np.random.choice([True, False], self.image_count, p=[self.training_split, 1.0-self.training_split])
        not_training_set = np.invert(training_set)
        self.training_images = [x for x,y in zip(self.images, training_set) if y]
        self.validation_images = [x for x,y in zip(self.images, not_training_set) if y]
        if self.label_dir:
            self.training_masks = [x for x,y in zip(self.masks, training_set) if y]
            self.validation_masks = [x for x,y in zip(self.masks, not_training_set) if y]

        self.steps_per_epoch = len(self.training_images)
        self.validation_steps = len(self.validation_images)
        self.epoch_index = 0
        self.validation_index = 0
        self.width = None
        self.height = None
        self.batch_size = None
        self.validation_batch_size = None

    def training_batch(self, batch_size=None):
        ''' if batch_size == None, then batch_size == epoch '''
        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = self.steps_per_epoch
        while True:
            image_list = []
            mask_list = []

            for i in range(self.batch_size):
                img = cv2.imread(self.training_images[self.epoch_index], cv2.IMREAD_COLOR)
                mask = cv2.imread(self.training_masks[self.epoch_index], cv2.IMREAD_GRAYSCALE)
                if self.width == None:
                    self.width = img.shape[1]
                if self.height == None:
                    self.height = img.shape[0]
                self.epoch_index += 1
                if self.epoch_index >= self.steps_per_epoch:
                    self.epoch_index = 0
                image_list.append(img)
                mask_list.append(mask)

            image_list = np.array(image_list, dtype=np.float32) #Note: don't scale input, because use batchnorm after input
            mask_list = np.array(mask_list, dtype=np.float32)
            mask_list /= 255.0 # [0,1]
            
            mask_list= mask_list.reshape(self.batch_size,self.height*self.width, 1) #NUMBER_OF_CLASSES
                    
            yield image_list, mask_list

    def validation_batch(self, validation_batch_size=None):
        ''' if batch_size == None, then batch_size == epoch '''
        if validation_batch_size:
            self.validation_batch_size = validation_batch_size
        else:
            self.validation_batch_size = self.validation_steps
        while True:
            image_list = []
            mask_list = []

            for i in range(self.validation_batch_size):
                img = cv2.imread(self.validation_images[self.validation_index], cv2.IMREAD_COLOR)
                mask = cv2.imread(self.validation_masks[self.validation_index], cv2.IMREAD_GRAYSCALE)
                if self.width == None:
                    self.width = img.shape[1]
                if self.height == None:
                    self.height = img.shape[0]
                self.validation_index += 1
                if self.validation_index >= self.validation_steps:
                    self.validation_index = 0
                image_list.append(img)
                mask_list.append(mask)

            image_list = np.array(image_list, dtype=np.float32) #Note: don't scale input, because use batchnorm after input
            mask_list = np.array(mask_list, dtype=np.float32)
            mask_list /= 255.0 # [0,1]
            
            mask_list= mask_list.reshape(self.validation_batch_size,self.height*self.width, 1) #NUMBER_OF_CLASSES
                    
            yield image_list, mask_list  

    def get_random_validation(self):
        index = randint(0, self.validation_steps)
        img = cv2.imread(self.validation_images[index], cv2.IMREAD_COLOR)
        if self.masks:
            mask = cv2.imread(self.validation_masks[index], cv2.IMREAD_GRAYSCALE)
        else:
            mask = None
        return img, mask  

    def get_image_and_mask(self, index):
        '''return the mask and image for an index '''
        img = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        if self.masks:
            mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        else:
            mask = None
        return img, mask

    def show_all(self):
        ''' overlay the mask onto the image '''
        for index in range(self.image_count):
            img, mask8 = self.get_image_and_mask(index)
            if mask8:
                mask = np.zeros_like(img)
                mask[:,:,2] = mask8
                alpha = 0.25
                cv2.addWeighted(mask, alpha, img, 1 - alpha, 0, img)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,self.images[index],(20,20), font, 0.4,(0,0,0),1,cv2.LINE_AA)
            cv2.imshow('img',img)
            key = cv2.waitKey(0)
            if key == 27: # esc
                break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    bg = batch_generator()
    bg.show_all()
    #im, mask = bg.get_random_validation()
