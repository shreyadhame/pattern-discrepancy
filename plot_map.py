#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots spatial map
"""

__title__ = ""
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreya.dhame@mpimet.mpg.de"

import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import proplot as pplt

ROBINSON_CENTRAL_LON = 180

class MidpointNormalize(mcolors.Normalize):
    """
    Normalise the colorbar so that diverging bars work their way either side from a prescribed midpoint value.
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def setup_map_plot(figsize=(7.48, 2.5), nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[1, 0.04], hspace=0.0, wspace=0.05, projection=ccrs.Robinson(ROBINSON_CENTRAL_LON)):
    """Sets up the figure and axes for map plots."""
    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, width_ratios=width_ratios, height_ratios=height_ratios, hspace=hspace, wspace=wspace)
    axs = [fig.add_subplot(spec[0, i], projection=projection) for i in range(ncols)]
    ax_cb = fig.add_subplot(spec[1, :])  # Colorbar axis
    return fig, axs, ax_cb

def plot_map_gridspec(ax, var, lon, lat, levels, cmap, mp=0, central_longitude=180,
                       extent=False, lat1=-90, lat2=90, lon1=0, lon2=360,
                       lat_step=10, lon_step=60, ticks=True, land=True,
                       title='Give Subplot Title Here', fontsize=12, pad=2, loc_title='left'):
    """
    Plot map using gridspec.
    """
    transform = ccrs.PlateCarree()
    ax.coastlines(lw=1.)
    
    if extent:
        ax.set_extent([lon1, lon2, lat1, lat2], crs=transform)

    if ticks:
        yticks = np.arange(lat1, lat2 + 1, lat_step)
        xticks = np.arange(lon1, lon2 + 1, lon_step)
        ax.set_xticks(xticks, crs=transform)
        ax.set_yticks(yticks, crs=transform)

        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

    vmin, vmax = levels[0], levels[-1]
    norm = MidpointNormalize(midpoint=mp, vmin=vmin, vmax=vmax)
    ax.contourf(lon, lat, var, levels=levels, transform=transform,
                cmap=cmap, extend='both', norm=norm)

    if land:
        ax.add_feature(cfeature.LAND, facecolor='#B1B1B1', linewidth=0.3)

    ax.set_title(title, pad=pad, loc=loc_title)

def create_colorbar(ax_cb, cmap, levels, label):
    """Creates the colorbar for the plot."""
    vmin, vmax = levels.min(), np.round(levels.max(), 2)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation='horizontal',
                                     extend='both', ticks=levels[::2])
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label(label=label, size=9)