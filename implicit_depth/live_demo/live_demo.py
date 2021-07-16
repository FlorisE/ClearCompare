#!/usr/bin/env python3


"""Live view of RGB-D LIDF."""

import argparse
import glob
import os
import shutil
import sys
import time

from attrdict import AttrDict

# from PIL import Image

import cv2

# import h5py

import numpy as np
# import numpy.ma as ma

from realsense import camera

import termcolor

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils  # noqa: E402 I100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run live demo of depth completion on realsense camera')
    parser.add_argument('-c', '--configFile', required=True,
                        help='Path to config yaml file',
                        metavar='path/to/config.yaml')
    args = parser.parse_args()

    # Initialize Camera
    print('Running live demo of depth completion.')
    print('Make sure realsense camera is streaming.\n')
    rcamera = camera.Camera()
    camera_intrinsics = rcamera.color_intr
    realsense_fx = camera_intrinsics[0, 0]
    realsense_fy = camera_intrinsics[1, 1]
    realsense_cx = camera_intrinsics[0, 2]
    realsense_cy = camera_intrinsics[1, 2]
    time.sleep(1)

    # Load Config File
    CONFIG_FILE_PATH = args.configFile
    with open(CONFIG_FILE_PATH) as fd:
        config_yaml = yaml.safe_load(fd)
    config = AttrDict(config_yaml)

    # Create directory to save captures
    runs = sorted(glob.glob(os.path.join(config.captures_dir, 'exp-*')))
    prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
    captures_dir = os.path.join(config.captures_dir,
                                'exp-{:03d}'.format(prev_run_id))
    if os.path.isdir(captures_dir):
        if len(os.listdir(captures_dir)) > 5:
            # Min 1 file always in folder: copy of config file
            captures_dir = os.path.join(config.captures_dir,
                                        'exp-{:03d}'.format(prev_run_id + 1))
            os.makedirs(captures_dir)
    else:
        os.makedirs(captures_dir)

    # Save a copy of config file in the logs
    shutil.copy(CONFIG_FILE_PATH, os.path.join(captures_dir, 'config.yaml'))

    print('Saving captured images to folder: ' +
          termcolor.colored('"{}"'.format(captures_dir), 'blue'))
    print('\n Press "c" to capture and save image, press "q" to quit\n')

    while True:
        color_img, input_depth = rcamera.get_data()
        input_depth = input_depth.astype(np.float32)

        # Display results
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2BGR)
        idm = utils.depth2rgb(input_depth,
                              min_depth=config.depthVisualization.minDepth,
                              max_depth=config.depthVisualization.maxDepth,
                              color_mode=cv2.COLORMAP_JET, reverse_scale=True),
        grid_image = np.concatenate((color_img, idm), 1)
        cv2.imshow('Live Demo', grid_image)
        keypress = cv2.waitKey(10) & 0xFF
        if keypress == ord('q'):
            break
        elif keypress == ord('c'):
            pass  # not implemented yet

    cv2.destroyAllWindows()
