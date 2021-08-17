#!/usr/bin/env python3


"""Live view of RGB-D LIDF."""

import argparse
import glob
import os
import os.path as osp
import shutil
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision import transforms
from imgaug import augmenters as iaa
from scipy.ndimage.measurements import label as connected_components

from attrdict import AttrDict

# from PIL import Image

import cv2

# import h5py

import numpy as np
# import numpy.ma as ma

from realsense import camera

import termcolor

import yaml

sys.path.append(
    os.path.join(os.path.dirname(__file__), '/workspace/implicit_depth/src'))
sys.path.append(
    os.path.join(os.path.dirname(__file__), '/workspace/ClearCompare/implicit_depth/live_demo/mask_network'))
from mask_network.modeling import deeplab

os.chdir('/workspace/implicit_depth/src')

import models.pipeline as pipeline # NOQA E402
from utils.training_utils import restore # NOQA E402
import utils.data_augmentation as data_augmentation



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


class PredictRefine(object):
    def __init__(self, opt):
        super(PredictRefine, self).__init__()
        self.opt = opt
        if self.opt.dist.ddp:
            print('Use GPU {} in Node {} for training'.format(
                self.opt.gpu_id, self.opt.dist.node_rank))
        # set device as local gpu id.
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_id))
        torch.cuda.set_device(self.device)
        if self.opt.dist.ddp:
            dist.barrier()
        self.setup_model()
        # sync all processes at the end of init
        if self.opt.dist.ddp:
            dist.barrier()

    def setup_model(self):
        print('===> Building models, GPU {}'.format(self.opt.gpu_id))
        self.lidf = pipeline.LIDF(self.opt, self.device)
        self.refine_net = pipeline.RefineNet(self.opt, self.device)

        if self.opt.lidf_ckpt_path is not None and \
                osp.isfile(self.opt.lidf_ckpt_path):
            loc = 'cuda:{}'.format(self.opt.gpu_id)
            checkpoint = torch.load(self.opt.lidf_ckpt_path, map_location=loc)
            restore(self.lidf.resnet_model, checkpoint['resnet_model'])
            restore(self.lidf.pnet_model, checkpoint['pnet_model'])
            restore(self.lidf.offset_dec, checkpoint['offset_dec'])
            restore(self.lidf.prob_dec, checkpoint['prob_dec'])
            print('Loaded checkpoint at epoch {} from {}.'.format(
                checkpoint['epoch'], self.opt.lidf_ckpt_path))
        else:
            raise ValueError('LIDF should be pretrained!')

        # freeze lidf
        for param in self.lidf.parameters():
            param.requires_grad = False

        # load checkpoint
        if self.opt.checkpoint_path is not None and \
                osp.isfile(self.opt.checkpoint_path):
            loc = 'cuda:{}'.format(self.opt.gpu_id)
            checkpoint = torch.load(self.opt.checkpoint_path, map_location=loc)
            restore(self.refine_net.pnet_model,
                    checkpoint['pnet_model_refine'])
            restore(self.refine_net.offset_dec,
                    checkpoint['offset_dec_refine'])
            print('Loaded checkpoint at epoch {} from {}.'.format(
                checkpoint['epoch'], self.opt.checkpoint_path))
        if self.opt.exp_type in ['test'] and self.opt.checkpoint_path is None:
            raise ValueError('Should identify checkpoint_path for testing!')

        # freeze refine_net
        for param in self.refine_net.parameters():
            param.requires_grad = False

        # ddp setting
        if self.opt.dist.ddp:
            # batchnorm to syncbatchnorm
            self.refine_net = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.refine_net)
            print('sync batchnorm at GPU {}'.format(self.opt.gpu_id))
            # distributed data parallel
            self.refine_net = nn.parallel.DistributedDataParallel(
                self.refine_net, device_ids=[self.opt.gpu_id],
                find_unused_parameters=True)
            print('DistributedDataParallel at GPU {}'.format(self.opt.gpu_id))

    def run_iteration(self, epoch, iteration, iter_len, exp_type, vis_iter, batch):
        pred_mask = None
        success_flag, data_dict, loss_dict = self.lidf(batch, exp_type, epoch, pred_mask)
        if not success_flag:
            print("Not successful")
            return None, None
        return self.refine_net(exp_type, epoch, data_dict)


    def predict(self, batch, epoch=0):
        with torch.no_grad():
            self.lidf.eval()
            self.refine_net.eval()
            data_dict, _ = self.run_iteration(epoch, 0, 1, 'predict', 99999, batch)
            if not data_dict:
                return None, None
            return data_dict

    def process(self, rgb_img, depth_img, camera_params, corrupt_mask):
        scale = (self.opt.dataset.img_width / rgb_img.shape[1], self.opt.dataset.img_height / rgb_img.shape[0])
        rgb_img = self.process_rgb(rgb_img)
        xyz_corrupt = data_augmentation.compute_xyz(depth_img, camera_params)
        xyz_corrupt_cv2 = cv2.resize(xyz_corrupt, (self.opt.xres, self.opt.yres), interpolation=cv2.INTER_NEAREST)
        xyz_corrupt = torch.from_numpy(xyz_corrupt_cv2).permute(2, 0, 1).float()

        # generate valid mask
        corrupt_mask = corrupt_mask.copy()
        corrupt_mask = self.process_label(corrupt_mask)
        corrupt_mask[corrupt_mask!=0] = 1
        corrupt_mask_float = torch.from_numpy(corrupt_mask).unsqueeze(0).float()
        corrupt_mask_label = torch.from_numpy(corrupt_mask).long()

        valid_mask = 1 - corrupt_mask
        valid_mask[depth_img==0] = 0
        valid_mask_float = torch.from_numpy(valid_mask).unsqueeze(0).float()
        valid_mask_label = torch.from_numpy(valid_mask).long()

        # Camera parameters
        camera_params['fx'] *= scale[0]
        camera_params['fy'] *= scale[1]
        camera_params['cx'] *= scale[0]
        camera_params['cy'] *= scale[1]

        sample = {
            'rgb': rgb_img,
            'depth_corrupt': depth_img,
            'xyz_corrupt': xyz_corrupt,
            'xyz_corrupt_cv2': xyz_corrupt_cv2,
            'fx': torch.tensor(camera_params['fx']),
            'fy': torch.tensor(camera_params['fy']),
            'cx': torch.tensor(camera_params['cx']),
            'cy': torch.tensor(camera_params['cy']),
            'corrupt_mask': corrupt_mask_float,
            'corrupt_mask_label': corrupt_mask_label,
            'valid_mask': valid_mask_float,
            'valid_mask_label': valid_mask_label,
            'item_path': 'RealSense'
        }
        return sample

    def process_rgb(self, rgb_img):
        rgb_img = cv2.resize(rgb_img, (self.opt.dataset.img_width, self.opt.dataset.img_height), interpolation=cv2.INTER_LINEAR)
        # BGR to RGB
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        # normalize by mean and std
        rgb_img = data_augmentation.standardize_image(rgb_img)
        rgb_img = data_augmentation.array_to_tensor(rgb_img) # Shape: [3 x H x W]

        return rgb_img

    def process_label(self, mask):
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        foreground_labels, num_components = connected_components(mask == 255)

        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)
        mapped_labels = foreground_labels.copy()
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels

        foreground_labels = cv2.resize(foreground_labels, (self.opt.xres, self.opt.yres), interpolation=cv2.INTER_NEAREST)

        return foreground_labels


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
    time.sleep(1)

    # Load Config File
    CONFIG_FILE_PATH = args.configFile
    with open(CONFIG_FILE_PATH) as fd:
        config_yaml = yaml.safe_load(fd)
    config = AttrDict(config_yaml)

    camera_params = {
        'fx': camera_intrinsics[0, 0],
        'fy': camera_intrinsics[1, 1],
        'cx': camera_intrinsics[0, 2],
        'cy': camera_intrinsics[1, 2],
        'xres': config.xres,
        'yres': config.yres
    }

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
    CHECKPOINT = torch.load(config.masks.pathWeightsFile, map_location='cpu')
    mask = deeplab.DeepLab(num_classes=config.masks.numClasses, backbone='drn',
                           sync_bn=True, freeze_bn=True)
    mask.load_state_dict(CHECKPOINT['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = mask.to(device)
    mask.eval()

    predict = PredictRefine(config)

    while True:
        color_img, input_depth = rcamera.get_data()
        color_img = cv2.resize(color_img, (config.xres, config.yres))
        transform = iaa.Sequential([
            iaa.Resize({
                "height": config.yres,
                "width": config.xres,
            }, interpolation='nearest'),
        ])
        det_tf = transform.to_deterministic()
        color_img_aug = det_tf.augment_image(color_img)
        color_img_array = np.array(color_img_aug)
        color_img_tensor = transforms.ToTensor()(color_img_array)
        color_img_tensor = torch.unsqueeze(color_img_tensor, 0)
        color_img_tensor = color_img_tensor.to(device)
        with torch.no_grad():
            mask_outputs = mask(color_img_tensor)

        predictions = torch.max(mask_outputs, 1)[1]
        predictions = predictions.squeeze(0).cpu().numpy()
        predicted_mask = np.zeros(predictions.shape, dtype=np.uint8)
        predicted_mask[predictions == 1] = 255

        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

        input_depth = cv2.resize(input_depth, (config.xres, config.yres))
        input_depth = input_depth.astype(np.float32)
        batch = predict.process(color_img, input_depth, camera_params, predicted_mask)
        batch['rgb'] = np.expand_dims(batch['rgb'], 0)
        batch['rgb'][0] = 1
        batch['rgb'] = torch.tensor(batch['rgb'])
        batch['xyz_corrupt'] = np.expand_dims(batch['xyz_corrupt'], 0)
        batch['xyz_corrupt'][0] = 1
        batch['xyz_corrupt'] = torch.tensor(batch['xyz_corrupt']) 
        batch['depth_corrupt'] = np.expand_dims(batch['depth_corrupt'], 0)
        batch['depth_corrupt'][0] = 1
        batch['depth_corrupt'] = torch.tensor(batch['depth_corrupt']) 
        batch['corrupt_mask'] = np.expand_dims(batch['corrupt_mask'], 0)
        batch['corrupt_mask'][0] = 1
        batch['corrupt_mask'] = torch.tensor(batch['corrupt_mask']) 
        batch['corrupt_mask_label'] = np.expand_dims(batch['corrupt_mask_label'], 0)
        batch['corrupt_mask_label'][0] = 1
        batch['corrupt_mask_label'] = torch.tensor(batch['corrupt_mask_label']) 
        batch['valid_mask'] = np.expand_dims(batch['valid_mask'], 0)
        batch['valid_mask'][0] = 1
        batch['valid_mask'] = torch.tensor(batch['valid_mask']) 
        batch['valid_mask_label'] = np.expand_dims(batch['valid_mask_label'], 0)
        batch['valid_mask_label'][0] = 1
        batch['valid_mask_label'] = torch.tensor(batch['valid_mask_label']) 
        result = predict.predict(batch)
        if result is None:
            continue

        # Display results
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        idm = depth2rgb(input_depth,
                        min_depth=config.depthVisualization.minDepth,
                        max_depth=config.depthVisualization.maxDepth,
                        color_mode=cv2.COLORMAP_JET, reverse_scale=True)
        grid_image = np.concatenate((color_img, idm, batch['xyz_corrupt_cv2']), 1)
        #grid_image2 = np.concatenate((batch['xyz_rgb']), 1)
        cv2.imshow('Live Demo', grid_image)
        #cv2.imshow('Live Demo 2', grid_image2)
        cv2.imshow('Live Demo 2', predicted_mask)
        print(f"Result: {result}")
        keypress = cv2.waitKey(10) & 0xFF
        if keypress == ord('q'):
            break
        elif keypress == ord('c'):
            pass  # not implemented yet

    cv2.destroyAllWindows()
