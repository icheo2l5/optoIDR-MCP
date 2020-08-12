# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
from pathlib import Path

# %%
from src import overlay
from src import helper

# %%
# %matplotlib inline
# %config InlineBackend.figure_format='retina'
import matplotlib.pyplot as plt
import matplotlib
plt.style.use("ggplot")
import seaborn as sns

# %%
# Prepare relative dirs
root_dir = Path(".").joinpath("..")
data_dir = root_dir.joinpath("data", "overlay")
report_dir = helper.mkdir_if_not_exist( root_dir.joinpath("report","overlay") )
figs = []

# %% [markdown]
# ### Load Data and preprocess

# %%
# Load data
df = overlay.load_data(data_dir)
df.head()

# %%
# Crop out droplet
align=False
df2 = overlay.crop_droplet(df, align=align, verbose=True)
df2.head()

# %%
# Take a look at the distribution of FWHM
fig = sns.catplot(y="cellid", x="fwhm", kind="swarm", data=df2.sort_values("cellid"))

# %%
# Filter out images with small droplet and resample remaining images
filter_fwhm = 7
df3 = overlay.resize_to_smallest(df2, filter_fwhm=filter_fwhm)

# %%
df3.head()

# %%
sns.catplot(y="cellid", x="fwhm", kind="swarm", data=df3.sort_values("cellid"))

# %%
fig = overlay.showall_single_channel(df3, "cropped", "droplet", "cropped_pos")
figs.append([fig, "cropped_droplets"])

# %%
fig = overlay.showall_single_channel(df3, "cropped", "mcp", "cropped_pos")
figs.append([fig, "cropped_mcps"])

# %% [markdown]
# ### Analysis
#
# 1. Average resized images

# %%
# Collect resized images in a single list for each channel
droplets = np.stack([row.resized.droplet.values for _,row in df3.iterrows()], axis=-1)
mcps = np.stack([row.resized.mcp.values for _,row in df3.iterrows()], axis=-1)

# Show averaged images
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
             f"from {df3.cellid.unique().size} cells")
fig.tight_layout(h_pad=2, w_pad=8)

figs.append([fig, "averaged_image"])

# %% [markdown]
# 2. Calculate radial profile by azimuthal averaging

# %%
# Azimuthal averaging
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
    
figs.append([fig, "radial_profile"])

# %% [markdown]
# ### Write useful results into report directory

# %%
# Set to TRUE only when running the entire notebook from scratch
to_save=True

# save averaged images as ImageJ compatible tif
import tifffile

if to_save:
    img = np.stack([np.mean(x, axis=-1) for x in [droplets,mcps]])
    img = img.astype(np.float32)
    fpath = report_dir.joinpath("averaged_image.tif")
    with tifffile.TiffWriter(str(fpath), imagej=True) as tif:
        for i in range(img.shape[0]):
            tif.save(img[i])

# save other created figures as svg files
if to_save:
    for fig,name in figs:
        fig.savefig(report_dir.joinpath(name+".svg"), dpi=300)
        
if to_save:
#     df3.to_csv(report_dir.joinpath("processed_images_info.csv"), index=False)
    df_rad.to_csv(report_dir.joinpath("radial_profile.csv"), index=False)

# %% [markdown]
# ### Always include session information

# %%
from sinfo import sinfo
sinfo(na=False, dependencies=True, write_req_file=False)

# %%
