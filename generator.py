import glob
import numpy as np
import cv2
from random import randint
from skimage import exposure
import imgaug as ia
from imgaug import augmenters as iaa


# def create_generators(image_dir=r'..\data\optical\062A\640x480\images',
#                       label_dir=r'..\data\optical\062A\640x480\masks',
#                       batch_size=1,
#                       target_size=(480, 640)):
#     ''' returns a tuple of training and validation ImageDataGenerators '''
#     from keras.preprocessing.image import ImageDataGenerator

#     data_gen_args = dict(
#         validation_split=0.15,
#         rescale=1.0 / 255,
#         # featurewise_center=True,
#         # featurewise_std_normalization=True,
#         #rotation_range=90.,
#         #width_shift_range=0.1,
#         #height_shift_range=0.1,
#         #zoom_range=0.2
#     )
#     image_datagen = ImageDataGenerator(**data_gen_args)
#     mask_datagen = ImageDataGenerator(**data_gen_args)

#     # Provide the same seed and keyword arguments to the fit and flow methods
#     seed = 1
#     # image_datagen.fit(image_dir, augment=True, seed=seed)
#     # mask_datagen.fit(label_dir, augment=True, seed=seed)

#     flow_args = dict(
#         batch_size=batch_size,
#         target_size=target_size,
#         # class_mode=None,
#         seed=seed)

#     image_generator = image_datagen.flow_from_directory(
#         image_dir, subset='training', **flow_args)

#     mask_generator = mask_datagen.flow_from_directory(
#         label_dir, subset='training', color_mode='grayscale', class_mode='binary', **flow_args)

#     validate_image_generator = image_datagen.flow_from_directory(
#         image_dir, subset='validation', **flow_args)

#     validate_mask_generator = mask_datagen.flow_from_directory(
#         label_dir, subset='validation', **flow_args)

#     # combine generators into one which yields image and masks
#     train_generator = zip(image_generator, mask_generator)
#     validate_generator = zip(validate_image_generator, validate_mask_generator)
#     return (train_generator, validate_generator)


class batch_generator(object):
    ''' generator for images and masks '''
    def __init__(self,
             image_dir=r'..\data\optical\062A\640x480\images\images\*.png',
             label_dir=None,
             training_split=0.8,
             scale_size=None,
             augment=False,
             shift=True):
        ''' scale_size is a tuple (W, H) '''

        self.training_split = training_split
        self.augment=augment
        self.shift = shift
        self.scale_size = scale_size
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
        self.fix_contrast = False

        # figure out the size of each image
        img = cv2.imread(self.images[0], cv2.IMREAD_COLOR)
        if self.scale_size:
            self.height = self.scale_size[0]
            self.width = self.scale_size[1]
        else:
            self.height = img.shape[0]
            self.width = img.shape[1]


        self.seq = iaa.Sequential([
            #iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-2, 2)
                ),
            iaa.AddToHueAndSaturation((-10, 10), name="AddToHueAndSaturation"),
            #iaa.Multiply((0.5, 1.5), per_channel=0.5, name="Multiply")
            # iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
        ])

        def activator_mask(images, augmenter, parents, default):
            if augmenter.name in ["AddToHueAndSaturation", "Multiply"]:
                return False
            else:
                # default value for all other augmenters
                return default
        self.hooks_activator_mask = ia.HooksImages(activator=activator_mask)





    def get_image_and_mask(self, index, source='training', augment=False):
        ''' return the optionally scaled image and mask from one of the 3 sets'''
        if source == 'training':
            img_src = self.training_images
            if self.label_dir:
                mask_src = self.training_masks
        elif source == 'validation':
            img_src = self.validation_images
            if self.label_dir:
                mask_src = self.validation_masks
        elif source == 'all':
            img_src = self.images
            if self.label_dir:
                mask_src = self.masks

        img = cv2.imread(img_src[index], cv2.IMREAD_COLOR)
        mask = None
        if self.label_dir:
            mask = cv2.imread(mask_src[index], cv2.IMREAD_GRAYSCALE)
        if self.scale_size:
            img = cv2.resize(img, self.scale_size)
            if self.label_dir:
                mask = cv2.resize(mask, self.scale_size)
        # if self.augment:
        #     #img = self.clahe.apply(img)
        #     img = cv2.equalizeHist(img)

        if augment:
            seq_det = self.seq.to_deterministic()
            img = seq_det.augment_image(img)
            if self.label_dir:
                mask = seq_det.augment_image(mask, hooks=self.hooks_activator_mask)

        return img, mask

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
                img, mask = self.get_image_and_mask(self.epoch_index, source='training', augment=False)

                self.epoch_index += 1
                if self.epoch_index >= self.steps_per_epoch:
                    self.epoch_index = 0
                image_list.append(img)
                mask_list.append(mask)

            seq_det = self.seq.to_deterministic()
            image_list = seq_det.augment_images(image_list)
            mask_list = seq_det.augment_images(mask_list, hooks=self.hooks_activator_mask)

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
                img, mask = self.get_image_and_mask(self.validation_index, source='validation', augment=False)

                self.validation_index += 1
                if self.validation_index >= self.validation_steps:
                    self.validation_index = 0
                image_list.append(img)
                mask_list.append(mask)

            seq_det = self.seq.to_deterministic()
            image_list = seq_det.augment_images(image_list)
            mask_list = seq_det.augment_images(mask_list, hooks=self.hooks_activator_mask)

            image_list = np.array(image_list, dtype=np.float32) #Note: don't scale input, because use batchnorm after input
            mask_list = np.array(mask_list, dtype=np.float32)
            mask_list /= 255.0 # [0,1]

            mask_list= mask_list.reshape(self.validation_batch_size,self.height*self.width, 1) #NUMBER_OF_CLASSES

            yield image_list, mask_list

    def get_random_validation(self):
        index = randint(0, self.validation_steps)
        return self.get_image_and_mask(index, source='validation')

    # -------------------------------------------------------
    # Image processing on optical images
    # -------------------------------------------------------
    def stretch_contrast(self, image, percentTop=3.0, percentBottom=0.5):
        ''' 
        stretch image contrast of BGR image by finding upper and lower bounds in histogram
        percent is percentage of pixels which are (bottom and below), or (top and above).
        This prevents random stuck pixels from messing up the stretch operation.
        '''
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, y = cv2.split(img)

        hist = cv2.calcHist([y], [0], None, [256], [0, 256])
        total_pix = float(y.size)
        fractionTop = percentTop / 100.0  # fraction of pixels
        fractionBottom = percentBottom / 100.0  # fraction of pixels
        cum_sum_up = np.cumsum(hist) / total_pix
        cum_sum_down = np.cumsum(hist[:, -1]) / total_pix
        # print cum_sum_up
        bottom = 0
        top = 255

        for j in range(256):
            if cum_sum_up[j] > fractionBottom:
                bottom = j
                break

        for j in range(256):
            if cum_sum_down[j] > fractionTop:
                top = 255 - j
                break
        #print total_pix, bottom, top
        #y = cv2.subtract(y, bottom)
        sf = 255.0 / (top)
        y = cv2.multiply(y, sf)
        #y = cv2.add(y, bottom)
        # y = y * (255.0 / (top - bottom) )
        #print np.min(y), np.max(y)

        # s = cv2.multiply(s, 1.8)
        img = cv2.merge((h, s, y))
        img_rgb_eq = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        return img_rgb_eq

    def show_all(self):
        ''' overlay the mask onto the image '''
        for index in range(self.image_count):
            img, mask8 = self.get_image_and_mask(index, source='all', augment=True)
            if mask8 is not None:
                mask = np.zeros_like(img)
                mask[:,:,2] = mask8
                alpha = 0.25
                cv2.addWeighted(mask, alpha, img, 1 - alpha, 0, img)
                # outline contour
                ret,thresh = cv2.threshold(mask8,127,255,0)
                im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (0,255,0), 3)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,self.images[index],(20,20), font, 0.4,(0,0,0),1,cv2.LINE_AA)
            cv2.imshow('img',img)
            key = cv2.waitKey(0)
            if key == 27: # esc
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    bg = batch_generator(
        image_dir=r"../data/optical/combined_training_set/images/*.png"
        , label_dir=r"../data/optical/combined_training_set/tissue_mask/*.png"
        )
    bg.show_all()
    #im, mask = bg.get_random_validation()
