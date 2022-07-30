# augment from kspace

"""
MRAugment applies channel-by-channel random data augmentation to MRI slices.
For example usage on the fastMRI and Stanford MRI datasets check out the scripts
in mraugment_examples.
"""
# dependency files:
# mraugment.helpers
#

import numpy as np
from math import exp
import torch
import torchvision.transforms.functional as TF
import fastmri
from mraugment.helpers import complex_crop_if_needed, crop_if_needed, complex_channel_first, complex_channel_last
from fastmri.data import transforms as T
from fastmri import fft2c, ifft2c, rss_complex, complex_abs
from matplotlib import pyplot as plt


def augment_image(im, transform_type = 'fliph', max_output_size=None):
    # Trailing dims must be image height and width (for torchvision)

    # initialization parameters:
    interp = True
    aug_max_translation_x = 0.125
    aug_max_translation_y = 0.08
    aug_max_rotation = 180
    aug_max_shearing_x = 15.0
    aug_max_shearing_y = 15.0
    aug_max_scaling = 0.25


    im = complex_channel_first(im)
    rng = np.random.RandomState()
    # ---------------------------
    # pixel preserving transforms
    # ---------------------------
    # Horizontal flip
    if transform_type == 'fliph':
        im = TF.hflip(im)

    # Vertical flip
    if transform_type == 'flipv':
        im = TF.vflip(im)

    # Rotation by multiples of 90 deg
    if transform_type == 'rot90':
        k = rng.randint(1, 4)
        im = torch.rot90(im, k, dims=[-2, -1])

    # Translation by integer number of pixels
    if transform_type == 'translation':
        h, w = im.shape[-2:]
        t_x = rng.uniform(- aug_max_translation_x,  aug_max_translation_x)
        t_x = int(t_x * h)
        t_y = rng.uniform(- aug_max_translation_y,  aug_max_translation_y)
        t_y = int(t_y * w)

        pad, top, left = get_translate_padding_and_crop(im, (t_x, t_y))
        im = TF.pad(im, padding=pad, padding_mode='reflect')
        im = TF.crop(im, top, left, h, w)

    # ------------------------
    # interpolating transforms
    # ------------------------
    interp = False

    # Rotation
    if transform_type == 'rotation':
        interp = True
        rot = rng.uniform(- aug_max_rotation,  aug_max_rotation)
    else:
        rot = 0.

    # Shearing
    if transform_type == 'shearing':
        interp = True
        shear_x = rng.uniform(-aug_max_shearing_x,  aug_max_shearing_x)
        shear_y =  rng.uniform(- aug_max_shearing_y,  aug_max_shearing_y)
    else:
        shear_x, shear_y = 0., 0.

    # Scaling
    if transform_type == 'scaling':
        interp = True
        scale =  rng.uniform(1 -  aug_max_scaling, 1 +  aug_max_scaling)
    else:
        scale = 1.

    # Upsample if needed
    # upsample = interp and  upsample_augment
    # if upsample:
    #     upsampled_shape = [im.shape[-2] *  upsample_factor, im.shape[-1] *  upsample_factor]
    #     original_shape = im.shape[-2:]
    #     interpolation = TF.InterpolationMode.BICUBIC if  upsample_order == 3 else TF.InterpolationMode.BILINEAR
    #     im = TF.resize(im, size=upsampled_shape, interpolation=interpolation)

    # Apply interpolating transformations
    # Affine transform - if any of the affine transforms is randomly picked
    if interp:
        h, w = im.shape[-2:]
        pad =  get_affine_padding_size(im, rot, scale, (shear_x, shear_y))
        im = TF.pad(im, padding=pad, padding_mode='reflect')
        im = TF.affine(im,
                       angle=rot,
                       scale=scale,
                       shear=(shear_x, shear_y),
                       translate=[0, 0],
                       interpolation=TF.InterpolationMode.BILINEAR
                       )
        im = TF.center_crop(im, (h, w))

    # ---------------------------------------------------------------------
    # Apply additional interpolating augmentations here before downsampling
    # ---------------------------------------------------------------------

    # Downsampling
    # if upsample:
    #     im = TF.resize(im, size=original_shape, interpolation=interpolation)

    # Final cropping if augmented image is too large
    if max_output_size is not None:
        im = crop_if_needed(im, max_output_size)

    # Reset original channel ordering
    im = complex_channel_last(im)

    return im

# generate target from augmented im
def im_to_target(im, target_size):
    # Make sure target fits in the augmented image
    cropped_size = [min(im.shape[-3], target_size[0]),
                    min(im.shape[-2], target_size[1])]

    if len(im.shape) == 3:
        # Single-coil
        target = complex_abs(T.complex_center_crop(im, cropped_size))
    else:
        # Multi-coil
        assert len(im.shape) == 4
        target = T.center_crop(rss_complex(im), cropped_size)
    return target

def get_translate_padding_and_crop(im, translation):
    t_x, t_y = translation
    h, w = im.shape[-2:]
    pad = [0, 0, 0, 0]
    if t_x >= 0:
        pad[3] = min(t_x, h - 1) # pad bottom
        top = pad[3]
    else:
        pad[1] = min(-t_x, h - 1) # pad top
        top = 0
    if t_y >= 0:
        pad[0] = min(t_y, w - 1) # pad left
        left = 0
    else:
        pad[2] = min(-t_y, w - 1) # pad right
        left = pad[2]
    return pad, top, left

def get_affine_padding_size(im, angle, scale, shear):
    """
    Calculates the necessary padding size before applying the
    general affine transformation. The output image size is determined based on the
    input image size and the affine transformation matrix.
    """
    h, w = im.shape[-2:]
    corners = [
        [-h/2, -w/2, 1.],
        [-h/2, w/2, 1.],
        [h/2, w/2, 1.],
        [h/2, -w/2, 1.]
    ]
    mx = torch.tensor(TF._get_inverse_affine_matrix([0.0, 0.0], -angle, [0, 0], scale, [-s for s in shear])).reshape(2,3)
    corners = torch.cat([torch.tensor(c).reshape(3,1) for c in corners], dim=1)
    tr_corners = torch.matmul(mx, corners)
    all_corners = torch.cat([tr_corners, corners[:2, :]], dim=1)
    bounding_box = all_corners.amax(dim=1) - all_corners.amin(dim=1)
    px = torch.clip(torch.floor((bounding_box[0] - h) / 2), min=0.0, max=h-1)
    py = torch.clip(torch.floor((bounding_box[1] - w) / 2),  min=0.0, max=w-1)
    return int(py.item()), int(px.item())

def plot_from_im(im):
    im_abs = complex_abs(im)   # Compute absolute value to get a real image
    plt.imshow(np.abs(im_abs.numpy()), cmap='gray')