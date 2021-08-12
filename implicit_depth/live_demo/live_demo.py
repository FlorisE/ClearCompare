#!/usr/bin/env python3


"""Live view of RGB-D LIDF."""

import argparse
import glob
import os
import shutil
import sys
import time

import torch

from attrdict import AttrDict

# from PIL import Image

import cv2

# import h5py

import numpy as np
# import numpy.ma as ma

from realsense import camera

import termcolor

import yaml

os.chdir('/workspace/implicit_depth/src')

sys.path.append(
    os.path.join(os.path.dirname(__file__), '/workspace/implicit_depth/src'))
from models.pipeline import LIDF # NOQA E402


def _normalize_depth_img(depth_img, dtype=np.uint8, min_depth=0.0,
                         max_depth=1.0):
    """Convert a floating point depth image to uint8 or uint16 image.

    The depth image is first scaled to (0.0, max_depth) and then scaled and
    converted to given datatype.
    Args:
        depth_img (numpy.float32): Depth image, value is depth in meters
        dtype (numpy.dtype, optional): Defaults to np.uint16. Output data type.
            Must be np.uint8 or np.uint16
        max_depth (float, optional): The max depth to be considered in the
            input depth image. The min depth is considered to be 0.0.
    Raises:
        ValueError: If wrong dtype is given
    Returns:
        numpy.ndarray: Depth image scaled to given dtype

    """
    if dtype != np.uint16 and dtype != np.uint8:
        msg = 'Unsupported dtype {}. Must be one of ("np.uint8", "np.uint16")'
        raise ValueError(msg.format(dtype))

    # Clip depth image to given range
    depth_img = np.ma.masked_array(depth_img, mask=(depth_img == 0.0))
    depth_img = np.ma.clip(depth_img, min_depth, max_depth)

    # Get min/max value of given datatype
    type_info = np.iinfo(dtype)
    max_val = type_info.max

    # Scale the depth image to given datatype range
    depth_img = ((depth_img - min_depth) / (max_depth - min_depth)) * max_val
    depth_img = depth_img.astype(dtype)

    # Convert back to normal numpy array from masked numpy array
    depth_img = np.ma.filled(depth_img, fill_value=0)

    return depth_img


def depth2rgb(depth_img, min_depth=0.0, max_depth=1.5,
              color_mode=cv2.COLORMAP_JET, reverse_scale=False,
              dynamic_scaling=False):
    """Generate RGB representation of a depth image.

    To do so, the depth image has to be normalized by specifying a min and max
    depth to be considered. Holes in the depth image (0.0) appear black in
    color.
    Args:
        depth_img (numpy.ndarray): Depth image, values in meters.
            Shape=(H, W), dtype=np.float32
        min_depth (float): Min depth to be considered
        max_depth (float): Max depth to be considered
        color_mode (int): Integer or cv2 object representing which coloring
            scheme to us. Please consult
            https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html
            Each mode is mapped to an int. Eg: cv2.COLORMAP_AUTUMN = 0.
            This mapping changes from version to version.
        reverse_scale (bool): Whether to make the largest values the smallest
            to reverse the color mapping
        dynamic_scaling (bool): If true, the depth image will be colored
            according to the min/max depth value within the
            image, rather that the passed arguments.
    Returns:
        numpy.ndarray: RGB representation of depth image. Shape=(H,W,3)

    """
    # Map depth image to Color Map
    if dynamic_scaling:
        dis = _normalize_depth_img(depth_img, dtype=np.uint8,
                                   min_depth=max(
                                       depth_img[depth_img > 0].min(),
                                       min_depth),
                                   max_depth=min(depth_img.max(), max_depth))
        # Added a small epsilon so that min depth does not show up as black
        # due to invalid pixels
    else:
        # depth image scaled
        dis = _normalize_depth_img(depth_img, dtype=np.uint8,
                                   min_depth=min_depth, max_depth=max_depth)

    if reverse_scale is True:
        dis = np.ma.masked_array(dis, mask=(dis == 0.0))
        dis = 255 - dis
        dis = np.ma.filled(dis, fill_value=0)

    depth_img_mapped = cv2.applyColorMap(dis, color_mode)
    depth_img_mapped = cv2.cvtColor(depth_img_mapped, cv2.COLOR_BGR2RGB)

    # Make holes in input depth black:
    depth_img_mapped[dis == 0, :] = 0

    return depth_img_mapped


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

    device = torch.device('cuda:{}'.format(config.gpu_id))
    torch.cuda.set_device(device)
    model = LIDF(config, device)
    model.eval()

    while True:
        color_img, input_depth = rcamera.get_data()
        input_depth = input_depth.astype(np.float32)

        model(color_img, exp_type='val', epoch=0)

        # Display results
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        idm = depth2rgb(input_depth,
                        min_depth=config.depthVisualization.minDepth,
                        max_depth=config.depthVisualization.maxDepth,
                        color_mode=cv2.COLORMAP_JET, reverse_scale=True)
        grid_image = np.concatenate((color_img, idm), 1)
        cv2.imshow('Live Demo', grid_image)
        keypress = cv2.waitKey(10) & 0xFF
        if keypress == ord('q'):
            break
        elif keypress == ord('c'):
            pass  # not implemented yet

    cv2.destroyAllWindows()
