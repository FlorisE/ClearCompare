"""Misc functions."""

# Completely based on ClearGrasp utils:
# https://github.com/Shreeyak/cleargrasp/

import cv2

import numpy as np


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
