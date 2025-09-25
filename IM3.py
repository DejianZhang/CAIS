import numpy as np
from scipy.ndimage import zoom
import h5py
from PIL import Image

def imresize3(volume, output_size=None, scale=None, method='cubic'):
    if output_size is None and scale is None:
        raise ValueError("Either output_size or scale must be specified.")
    
    if scale is not None:
        if isinstance(scale, (int, float)):
            scale = (scale, scale, scale)
        elif len(scale) != 3:
            raise ValueError("Scale must be a scalar or a 3-element tuple.")
    
    if output_size is not None:
        if len(output_size) != 3:
            raise ValueError("Output size must be a 3-element tuple.")
        scale = (
            output_size[0] / volume.shape[0],
            output_size[1] / volume.shape[1],
            output_size[2] / volume.shape[2]
        )
    
    if method == 'cubic':
        order = 3
    elif method == 'linear':
        order = 1
    elif method == 'nearest':
        order = 0
    else:
        raise ValueError("Unsupported interpolation method.")

    resized_volume = zoom(volume, scale, order=order)
    return resized_volume




def imresize2(image, output_size, method='bicubic', antialiasing=True):
    """
    Resize a 2D image similar to MATLAB's imresize.
    
    Parameters:
    - image: Input image as a NumPy array or PIL Image.
    - output_size: Desired output size as a tuple (rows, cols).
    - method: Interpolation method ('nearest', 'bilinear', 'bicubic').
    - antialiasing: Whether to apply antialiasing when reducing image size.
    
    Returns:
    - resized_image: Resized image as a NumPy array.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Check output size
    if len(output_size) != 2:
        raise ValueError("output_size must be a tuple of (rows, cols).")

    # Choose interpolation method
    if method == 'nearest':
        resample_method = Image.NEAREST
    elif method == 'bilinear':
        resample_method = Image.BILINEAR
    elif method == 'bicubic':
        resample_method = Image.BICUBIC
    else:
        raise ValueError("Unsupported interpolation method.")

    # Resize the image
    resized_image = image.resize(output_size, resample=resample_method)

    return np.array(resized_image)




