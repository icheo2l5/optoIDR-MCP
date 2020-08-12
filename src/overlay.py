'''overlay.py

Module to average images within the entire dataset.

This module provides functions to process and analyze images in the dataset, 
including segmentation and aligning of circular droplets, and averaging the 2D 
images and the resulting radial profile.

**Note**
The data files need to be inside `../data/overlay/<folder>` with individual 
tiff file names satisfying the convention `<cellid>-<roiid>` where...
* `<folder>` is `d`: a one-digit number
* `<cellid>` is `dddd-dd`: four digits followed by two digits
* `<roiid>` is `d*`: any number
'''

import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from src import helper
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns
from functools import partial, reduce
from skimage import feature, filters, transform



def main():
    """Perform analysis and save results when run as a script."""

    # PARAMETERS TO CHANGE
    align = False
    filter_fwhm = 7
    channels = ["droplet", "mcp"]

    # Prepare relative dirs
    root_dir = Path(".")
    data_dir = root_dir.joinpath("data", "overlay")
    report_dir = helper.mkdir_if_not_exist(
        root_dir.joinpath("report", "overlay"))
    figs = []

    # Load data
    df = load_data(data_dir)
    print(f"Successfully loaded data from {df.shape[0]} tiff files.")

    # Crop out droplet
    df2 = crop_droplet(df, align=align)

    # Filter out images with small droplet and resample remaining images
    df3 = resize_to_smallest(df2, filter_fwhm=filter_fwhm)

    # Save all identified droplet and mcp images
    figs += [[showall_single_channel(df3, "cropped", ch,
                                     pos_name="cropped_pos"),
              "cropped_"+ch]
             for ch in channels]

    # Average images
    figs += [[average_resized_images(df3, align=align, filter_fwhm=filter_fwhm),
             "averaged_image"]]
    
    # Azimuthal averaging
    figs += [[radial_profile(df3), "radial_profile"]]

    # Save figures
    for fig,name in figs:
        fig.savefig(report_dir.joinpath(name+".svg"), dpi=300)


# ==========
# Functions at dataset level
# ==========

def load_data(data_dir):
    """Load image datasets from tiff files and build dataframe for all files.
    
    """

    # build list of files
    p = data_dir.glob("**/*.tif")
    files =  [x for x in p if x.is_file()]

    # Construct channel layout info for tiff files in foler 1 and 2
    channels = pd.read_json(data_dir.joinpath("info.JSON"),
                            orient="records", dtype=False)

    # Build dataframe for file info
    df = (pd.DataFrame({"path": [str(f) for f in files],
                        "filename": [f.name for f in files]})
        .assign(
            folder = lambda x: x.path.str.extract(r"(\d)/\d{4}-\d{2}"),
            cellid = lambda x: x.filename.str.extract(r"(\d{4}-\d{2})"),
            roiid = lambda x: x.filename.str.extract(r"\d{4}-\d{2}-(\d*)"))
        .merge(channels, on="folder")
    )

    # Load in data
    df["img"] = df.apply(
        lambda x: helper.ds_from_tiff(x["path"], x["channels"]),
        axis=1)

    # Convert pixel value to [0,1]
    # AiryScan has 16-bit camera
    bit_depth = 16
    df.img = df.img / (2**bit_depth-1)

    # Sort rows according to filename
    df = (df.sort_values(by="filename")
            .reset_index().drop("index", axis="columns"))

    return df



def crop_droplet(df, align=True, verbose=True):
    """Segment out droplet and find droplet's FWHM for each dataset.
    
    """

    # Identify blobs in each image
    df_ = df.apply(_ds_find_droplets, axis="columns")

    # Filter out images with more than one blob and then find FWHM for the droplet
    # Sort images by decreasing FWHM of the droplet
    df2 = (df_
        .query(" n_droplet == 1 ")
        .apply(_ds_fit_fwhm, axis="columns")
        .dropna(axis="rows", how="any")
        .sort_values(by="fwhm", ascending=False)
    )

    # Rotate (align) if specified, then crop out a box with side length 2*FWHM
    if align:   
        df3 = (df2
            .apply(_ds_align, axis="columns").dropna(axis="rows", how="any")
            .apply(_ds_crop_aligned, axis="columns").dropna(axis="rows", how="any")
        )
    else:
        df3 = df2.apply(_ds_crop, axis="columns").dropna(axis="rows", how="any")

    # Print messages explaining what happened
    if verbose:
        print(f"After droplet segmentation, ...\n" +
              f"    {df2.shape[0]} images remained.\n" +
              f"    {df.shape[0]-df2.shape[0]} images thrown out because\n" +
              f"        more than one droplets identified, OR\n" + 
              f"        couldn't fit circular gaussian to the droplet.")
        print(f"After rotating and cropping, ...\n" +
              f"    {df3.shape[0]} images remained.\n" +
              f"    {df2.shape[0]-df3.shape[0]} images thrown out because:\n" +
              f"        Droplet too close to the edge of image,\n" + 
              f"        which might introduce empty pixels into further analysis.")

    df_out = df3
    return df_out



def resize_to_smallest(df, filter_fwhm=None, verbose=True):
    """Remove datasets with small droplet and resize images to the smallest.
    
    """

    # Filter out images with small droplet
    if filter_fwhm is not None:
        df2 = df.query(" fwhm >= @filter_fwhm ")
    else:
        df2 = df.copy()

    # Resample each image to fit the smallest one
    minshape = np.asarray(df2.cropped.apply(lambda x: [x.dims['x'], x.dims['y']]).min())
    resize = partial(_ds_resize, minshape=minshape)
    df_out = df2.apply(lambda x: resize(x), axis="columns")

    # Print messages explaining what happened
    if verbose:
        print(f"After filtering out small droplets, ...\n" +
              f"    {df_out.shape[0]} images remained.\n" +
              f"    {df.shape[0]-df_out.shape[0]} images thrown out.\n" +
              f"    Filtering criteria: FWHM >= {filter_fwhm} pixels.")
        print(f"Prepared resampled images as an xr.Dataset " + 
              f"stored in column `resized`, ...\n" +
              f"    Image dimensions: x: {minshape[1]}, y: {minshape[0]}.")

    return df_out



def showall_single_channel(df, ds_name, channel,
                           pos_name=None,
                           wrap=10,
                           cmap_name='viridis',
                           sharecolor=False,
                           sharexy=False,
                           title=True):
    """Show all images in dataframe with specified channel and droplet outline.
    
    Parameters
    ----------
    df : pandas.DataFrame
    ds_name : str
        Name of column (in `df`) where xr.Dataset is stored and to be shown.
    pos_name : str
        Name of column (in `df`) where center of droplet is stored.
    sharecolor : bool
    sharexy : bool
    title : bool
    wrap : int
    cmap_name : str
    
    Return
    ------
    matplotlib.Figure
    
    """
    
    cmap = plt.get_cmap(cmap_name)
    
    # Determine max value for color mapping
    if sharecolor:
        max_ = (df[ds_name]
            .apply(lambda x: x[channel].values.ravel().max()) # max for an img
            .max())
    
    
    # show droplet raw image and overlaid droplet outline
    fig, ax = helper.subplots_wrapcol(df.shape[0], axsize=(3,3),
                                   wrap=wrap, sharex=sharexy, sharey=sharexy)
    ax = ax.ravel()
    
    
    for (_,row),a in zip(df.iterrows(),ax):
        # show images using same contrast if specified
        if sharecolor:
            a.imshow(row[ds_name][channel].values, cmap=cmap,
                     vmin=0, vmax=max_)
        else:
            a.imshow(row[ds_name][channel].values, cmap=cmap)
            
        # draw outline of droplet if specified
        if pos_name is not None:
            c = plt.Circle(row[pos_name][::-1], row.fwhm/2,
                           color="white", linewidth=2, fill=False)
            a.add_patch(c)

        # Configure axis
        a.set_title(row.filename)

    # Axis off for all including unused trailing axes
    for a in ax:
        a.set_axis_off()
    
    fig.tight_layout(pad=0.5)
        
    return fig



def average_resized_images(df, align, filter_fwhm):
    # Collect resized images in a single list for each channel
    droplets = np.stack([row.resized.droplet.values
                         for _,row in df.iterrows()], axis=-1)
    mcps = np.stack([row.resized.mcp.values
                     for _,row in df.iterrows()], axis=-1)

    fig, ax = plt.subplots(ncols=2, figsize=(8,4))
    ax = ax.ravel()
    im1 = ax[0].imshow(np.mean(droplets, axis=-1))
    ax[0].set_axis_off()
    ax[0].set_title("Droplet")
    helper.colorbar(im1)
    im2 = ax[1].imshow(np.mean(mcps, axis=-1))
    ax[1].set_axis_off()
    ax[1].set_title("MCP")
    helper.colorbar(im2)

    fig.suptitle(f"Averaged image (aligned to max = {align}, shape = {droplets.shape[:-1]})\n" + 
                f"n={droplets.shape[-1]} droplets " + 
                f"with FWHM > {filter_fwhm} px " + 
                f"from {df.cellid.unique().size} cells")
    fig.tight_layout(h_pad=2, w_pad=8)

    return fig



def radial_profile(df):
    # Collect resized images in a single list for each channel
    droplets = np.stack([row.resized.droplet.values
                         for _,row in df.iterrows()], axis=-1)
    mcps = np.stack([row.resized.mcp.values
                     for _,row in df.iterrows()], axis=-1)

    profiles = [helper.radial_profile(np.mean(x, axis=-1))
                for x in [droplets, mcps]]
    profiles = [profile / np.nanmax(profile) for profile in profiles]
    channels = ["droplet", "mcp"]
    df_rad = pd.DataFrame.from_dict(dict(zip(channels,profiles)))
    df_rad = (df_rad
        .assign(bin=df_rad.index)
        .melt(id_vars=["bin"], value_vars=["droplet","mcp"],
            var_name="channel", value_name="RelativeIntensity"))

    with matplotlib.style.context("seaborn-ticks"):
        fig, ax = plt.subplots(figsize=(3,3))
        data = df_rad[df_rad["channel"]=="droplet"]
        ax.plot(data.bin, data.RelativeIntensity, label="Droplet", marker='o', markersize=5)
        data = df_rad[df_rad["channel"]=="mcp"]
        ax.plot(data.bin, data.RelativeIntensity, label="MCP", marker='o', markersize=5)
        ax.legend()
        ax.set_xlabel("Radial bin")
        ax.set_ylabel("Relative intensity")
        ax.set_ylim((-0.1,1.1))

    return fig


# ==========
# Private functions
# ==========

def _ds_find_droplets(df):
    img = df.img.droplet.values
    blobs, rads = helper.find_blobs(img)
    n = blobs.shape[0]
    df["n_droplet"] = n
    if n == 1:
        df["pos"] = blobs.ravel()
        df["rad"] = rads.item()
    else:
        df["pos"] = blobs
        df["rad"] = rads.ravel()
    return df

def _ds_fit_fwhm(df):
    try:
        df["fwhm"] = helper.get_fwhm(df.img.droplet.values, df.pos)
    except:
        df["fwhm_droplet"] = np.nan
        raise
    return df
    
def _ds_align(df):
    df["aligned"], df["aligned_pos"] = (
        _align(df.img, df.pos, df.fwhm))
    return df

def _ds_crop_aligned(df):
    try:
        df["cropped"], df["cropped_pos"] = (
            helper.crop(df.aligned, df.aligned_pos, 2*df.fwhm))
    except ValueError:
        df["cropped"] = np.nan
        df["cropped_pos"] = np.nan
    return df

def _ds_crop(df):
    try:
        df["cropped"], df["cropped_pos"] = (
            helper.crop(df.img, df.pos, 2*df.fwhm))
        df["cropped_dim"] = df["cropped"].dims['x']
    except ValueError:
        df["cropped"] = np.nan
        df["cropped_pos"] = np.nan
        df["cropped_dim"] = np.nan
    return df

def _ds_resize(df, minshape):
    func = partial(transform.resize, output_shape=minshape)
    resized_droplet = func(df.cropped.droplet.values)
    resized_mcp = func(df.cropped.mcp.values)
    df["resized"] = xr.Dataset(
        {
            'droplet': (['y','x'], resized_droplet),
            'mcp':     (['y','x'], resized_mcp)
        }
    )
    return df



def _align(ds, center, fwhm, r=1.5, check_far=True):
    """
    Rotate images in a dataset, with respect to the center of the blob such 
    that pixel of max sits on (+) X-axis.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset that contains 'droplet' and 'mcp' channels.
    center: tuple of float, length=2
        Center of the blob. Represented as (y, x).
    fwhm: float
        FWHM of circular 2D Gaussian fitted to the blob.

    Return
    ------
    np.ndarray
        Rotated image. Shape might be different (larger in general) from 
        the original image.
    """

    # Check image boundary is far enough from the droplet by attempting to 
    # crop a square with its side length 2*sqrt(2)*FWHM 
    if check_far:
        try:
            helper.crop(ds, center, 2*np.sqrt(2)*fwhm)
        except ValueError:
            # raise ValueError("Rotated image would contain fake values when " +
            #                  "being further cropped.")
            return np.nan, np.nan

    rotator = helper._rotate_max_to_xaxis(ds.mcp.values, center, r*fwhm)
    # Rotate images
    rotated = xr.Dataset(
        {
            'droplet': (['y','x'], rotator(ds.droplet.values)),
            'mcp':     (['y','x'], rotator(ds.mcp.values))
        }
    )
    # Get new center
    y0, x0 = np.rint(center).astype(int)
    center_matrix = np.full(ds.droplet.shape, False, dtype=bool)
    center_matrix[y0,x0] = True
    rotated_center_matrix = rotator(center_matrix)
    rotated_center = np.unravel_index(rotated_center_matrix.argmax(),
                                      rotated_center_matrix.shape)

    return rotated, np.array(rotated_center)

# End of private functions



if __name__ == "__main__":
    main()