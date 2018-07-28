from __future__ import print_function

import os
import sys
import argparse
import cv2
import numpy as np

# Resamples images and masks as appropriate for a given DNN model

def prepare_images(args):
    parent_parser = argparse.ArgumentParser(description='Resampler for images and masks.')

    parent_parser.add_argument('-i', '--image_directory', help='the image directory to process.',
                               metavar="", default=r'D:\data\optical\0148A_preloading')
    parent_parser.add_argument('-m', '--mask_directory', help='the image directory to process.',
                               metavar="", default=r'C:\data\002327\0')
    parent_parser.add_argument('-o', '--output_directory', help='output directory',
                               metavar="", default=r'C:\data\output')
    parent_parser.add_argument('-n', '--output_mask_directory', help='output mask directory',
                               metavar="", default=r'C:\data\output_mask')
                               
    parent_parser.add_argument('-w', '--width', type=int, default=640, metavar="",
                               help='output file width')
    parent_parser.add_argument('-v', '--height', type=int, default=480, metavar="",
                               help='output file height')
    parent_parser.add_argument('-x', '--x_offset', type=int, default=708, metavar="",
                               help='x_offset in source image')
    parent_parser.add_argument('-y', '--y_offset', type=int, default=192, metavar="",
                               help='x_offset in source image')
    parent_parser.add_argument('-j', '--src_width', type=int, default=640, metavar="",
                               help='source width in source image')
    parent_parser.add_argument('-k', '--src_height', type=int, default=480, metavar="",
                               help='source height in source image')

    args = parent_parser.parse_args(args)

    if args.mask_directory == 'None': #zork!
        args.mask_directory = None

    try:
        os.mkdirs(args.output_directory)
    except:
        pass

    if args.mask_directory:
        try:
            os.mkdirs(args.output_mask_directory)
        except:
            pass

    image_root = args.image_directory
    mask_root = args.mask_directory

    for root, _dirs, files in os.walk(image_root):
        sorted_files = sorted(files, reverse=False)
        for index, filename in enumerate(sorted_files):
            if filename.endswith(('.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                # if filename < begin_file or filename > end_file:
                #     continue
                image_path = os.path.join(image_root, filename)
                out_image_path = os.path.join(args.output_directory, filename).replace("\\","/")
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = image[args.y_offset:args.y_offset+args.src_height, args.x_offset:args.x_offset+args.src_width, :]
                image = cv2.resize(image, (args.width, args.height))
                cv2.imwrite(out_image_path, image)

                if mask_root:
                    mask_path = os.path.join(mask_root, filename).replace("\\","/")
                    out_mask_path = os.path.join(args.output_mask_directory, filename)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAY)
                    mask = mask[args.y_offset:args.y_offset+args.src_height, args.x_offset:args.x_offset+args.src_width]
                    mask = cv2.resize(mask, (args.width, args.height))
                    cv2.imwrite(out_mask_path, mask)


if __name__=='__main__':
    prepare_images(sys.argv[1:])
