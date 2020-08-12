import pathlib
import tifffile
import numpy as np
from scipy import optimize as opt
import xarray as xr
from functools import partial, reduce
from skimage import feature, filters, transform
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ==========
# FileIO
# ==========

def mkdir_if_not_exist(path):
    """Make dir if not existing. Return pathlib.Path for resulting dir."""
    
    _path = pathlib.Path(path) if isinstance(path,str) else path
    if not isinstance(_path, pathlib.Path):
        raise TypeError("path must be of type str or pathlib.Path")
    
    try:
        _path.mkdir(exist_ok=False)
    except FileExistsError:
        pass
    
    return _path



# ==========
# ImageIO
# ==========

def ds_from_tiff(file, channels):
    """Load tiff image and ruturn as an xarray.Dataset"""
    
    if isinstance(file, pathlib.Path):
        file = str(file)

    with tifffile.TiffFile(file) as tif:
        data = tif.asarray()
        das = _das_from_ndarray(data, channels)

    return xr.Dataset(dict(zip(channels, das)))

def _das_from_ndarray(arr, channels):
    """Wrap np.ndarray into a xarray.Dataset with channel info."""
    das = [xr.DataArray(a,
                        dims=["y", "x"],
                        name=ch)
           for a,ch in zip(arr, channels)]

    return das



# ==========
# Fitting
# ==========

# Adapted from
# 1. https://gist.github.com/nvladimus/fc88abcece9c3e0dc9212c2adc93bfe7
# 2. https://stackoverflow.com/a/26864260
# Note the following two have been modified to fit to a circular 2D gaussian

def _twoD_GaussianScaledAmp(data, xo, yo, sigma, amplitude, offset):
    """Function to fit, returns 2D gaussian function as 1D array"""
    (x, y) = data
    xo = float(xo)
    yo = float(yo)
    sigma_x = sigma
    sigma_y = sigma
    g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))
    return g.ravel()

def get_fwhm(img, pos_center):
    """Get FWHM(x,y) of a blob by 2D gaussian fitting
    Parameter:
        img - image as numpy array
    Returns: 
        FWHMs in pixels, along x and y axes.
    """
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    x, y = np.meshgrid(x, y)
    #Parameters: xpos, ypos, sigmaX, sigmaY, amp, baseline
    initial_guess = (pos_center[1],pos_center[0],10,1,0)
    # subtract background and rescale image into [0,1], with floor clipping
    bg = np.percentile(img,5)
    img_scaled = np.clip((img - bg) / (img.max() - bg),0,1)
    popt, pcov = opt.curve_fit(
        _twoD_GaussianScaledAmp, (x,y),
        img_scaled.ravel(),
        p0=initial_guess,
        bounds=((pos_center[1]-img.shape[1]*0.1,
                 pos_center[0]-img.shape[0]*0.1,
                 1, 0.5, -0.1),
                (pos_center[1]+img.shape[1]*0.1,
                 pos_center[0]+img.shape[0]*0.1,
                 (img.shape[1]+img.shape[0])/4, 1.5, 0.5)
               )
    )
    # xcenter, ycenter, sigma, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4]
    sigma = popt[2]
    FWHM = np.abs(4*sigma*np.sqrt(-0.5*np.log(0.5)))
    
    return FWHM


# Radial profile related tools from:
# https://github.com/keflavich/image_tools/blob/master/image_tools/radialprofile.py

# https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
def radial_profile(data, center=None):
    y, x = np.indices((data.shape))
    
    if center is None:
        center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
    
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 


# ==========
# Image processing
# ==========

def find_blobs(img):
    """Find blobs using Laplacian of Gaussian (LoG) method
    **NEED UPDATE**
    Return
    ------
    (n, 3) ndarray of float
        Each row represents (r, c, sigma) where r and c describe the 
        coordinate values for the blob center, and the sigma is the standard
        deviation of the isotropic Gaussian kernel which detected the blob.
        n is the number of blobs detected which ideally should be just one.
    
    """
    
    # max for determining threshold
    max_ = img.flatten().max()
    # Try threshold from high to low (max_ to 0.1*max_)
    ths = max_ * np.arange(1, 0, -0.1)
    for th in ths:
        blobs = feature.blob_log(img, threshold=th)
        if blobs.shape[0] == 1: break
    # transform sigma used to radius of blob
    # r ~ sqrt(2)*sigma for 2D image
    rad = blobs[:,2] * np.sqrt(2)
    
    return blobs[:,0:2], rad.flatten()



def _rotate_max_to_xaxis(img, center, width):
    """
    Rotate image w.r.t. center of blob s.t. pixel of max sits on (+) X-axis.

    Parameters
    ----------
    img: np.ndarray, shape = (m,n)
    center: tuple of float

    Return
    ------
    np.ndarray
        Rotated image. Shape might be different (larger in general) from 
        the original image.
    """

    # Prefilter image before identifying where max is
    img_ = filters.gaussian(img, sigma=1)


    # Search pixel with maximal value within a square for filtered image
    # (center at blob center, side length is width)
    w = width
    # Get index boundary. Make sure to round to int for indexing.
    y0, x0 = np.rint(center).astype(int)
    xmin, xmax = np.rint(x0 + np.array([-w/2, w/2])).astype(int)
    ymin, ymax = np.rint(y0 + np.array([-w/2, w/2])).astype(int)
    # Generate masked image and use it to find max
    mask = np.full(img.shape, False, dtype=bool)
    mask[ymin:ymax, xmin:xmax] = True
    masked = img_ * mask
    y, x = np.unravel_index(masked.argmax(), masked.shape)
    # find the angle of vector (x,y) relative to (x0,y0)
    phi = np.arctan2(y-y0, x-x0) * 180 / np.pi
    # Transformation function to use later
    rotator = partial(transform.rotate,
                      angle=phi, center=(y0,x0), resize=True, mode='constant')

    return rotator



def rotate_max_to_xaxis(img, center, width):
    rotator = _rotate_max_to_xaxis(img, center, width)
    
    return rotator(img)



def crop(ds, center, width):
    """Crop out a square subregion at specified center and width.
    
    """

    boundary_idx = (reduce(np.add,
        [np.broadcast_to(mat, (2,2))
         for mat in [center,
                     width/2*np.array([-1, 1]).reshape((2,1)),
                     np.array([0, 1]).reshape((2,1))
                    ]
        ]))
    boundary_idx = np.rint(boundary_idx).astype(int)
    img_boundary_idx = np.array([[0, 0],
                                 [ds.dims['y']-1, ds.dims['x']-1]])
    # The following is bad...
    if np.any(((img_boundary_idx[0,:] - boundary_idx[0,:])>=0) |
              ((img_boundary_idx[1,:] - boundary_idx[1,:])<0)):
        raise ValueError("Attempt to crop outside of image boundary.")

    cropped = ds.isel(x=slice(*boundary_idx[:,1]),
                      y=slice(*boundary_idx[:,0]))
    
    cropped_center = (boundary_idx[1,:] - boundary_idx[0,:] - 1)/2

    return cropped, cropped_center


# =========
# Tools for plotting
# =========

def colorbar(mappable):
    """Create and place colorbar nicely for the last matplotlib.Axes
    
    Adapted from https://joseph-long.com/writing/colorbars/
    
    """

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)

    return cbar



def subplots_wrapcol(n, wrap, axsize=(2,2), **kwargs):
    ncols = min(n, wrap)
    nrows = 1 + (n-1)//wrap
    figsize = (axsize[0] * ncols, axsize[1] * nrows)
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, **kwargs)

    return fig, ax