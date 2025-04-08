#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First and second low frequency patterns.
"""

__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"
#============================================================
import os
import sys
import argparse
import gc
import klepto
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import proplot as pplt
import cartopy.crs as ccrs
import matplotlib as mpl
from plot_maps import plot_maps_gridspec
from import_highresmip_data import ersst_tos, cobe_tos

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set global plot settings
plt.rc('font', size=12)
pplt.rc.update(grid=False)

#============================================================
def load_data(model):
    """Load data from klepto archive."""
    db = klepto.archives.dir_archive('LFCA_data', serialized=True, cached=False)
    return (
        db[f'{model}_pattern1'],
        db[f'{model}_pattern2'],
        db[f'{model}_lfcs']
    )
def create_figure():
    """Create the main figure and axes."""
    fig = plt.figure(figsize=(7.48, 4.48))
    widths = [1, 1]
    heights = [1, 0.05, 0.3, 0.6]
    spec = gridspec.GridSpec(ncols=2, nrows=4, width_ratios=widths,
                             height_ratios=heights, hspace=0.05, wspace=0.05)
    projection = ccrs.Robinson(180)
    
    ax1 = fig.add_subplot(spec[0, 0], projection=projection)
    ax2 = fig.add_subplot(spec[0, 1], projection=projection)
    ax3 = fig.add_subplot(spec[3, 0])
    ax4 = fig.add_subplot(spec[3, 1])
    ax_cb1 = fig.add_subplot(spec[1, :])
    
    return fig, (ax1, ax2, ax3, ax4, ax_cb1)

def plot_maps(axs, map_plots, lon_axis, lat_axis):
    """Plot spatial maps."""
    clev = np.arange(-0.4, 0.45, 0.05)
    titles = ['(b) Pattern 1', '(c) Pattern 2']
    
    for i, (ax, map_plot) in enumerate(zip(axs, map_plots)):
        plot_maps_gridspec(ax, map_plot, lon_axis, lat_axis, levels=clev, mp=0.,
                           cmap=pplt.Colormap('ColdHot'),
                           ticks=False, land=True, title=titles[i], loc_title='center', pad=5)
        
        plot_regions(ax)

def plot_regions(ax):
    """Plot specific regions on the map."""
    regions = [
        ([360, 0, 0, 360, 360], [-75, -75, -45, -45, -75], '--', 0.3),  # Southern Ocean
        ([-95, 180, 180, -95, -95], [-5, -5, 5, 5, -5], '-', 0.6),  # EP
        ([110, 180, 180, 110, 110], [-5, -5, 5, 5, -5], '-', 0.6),  # WP
        ([-70, -70, -160, -70, -70], [-45, -45, -5, -5, -45], '--', 0.3),  # southeastern Pacific warm bias
        ([-145, -110, -110, -145, -145], [10, 10, 30, 30, 10], '--', 0.3)  # Northeastern Pacific warm bias
    ]
    
    for lon, lat, style, width in regions:
        ax.plot([x % 360 for x in lon], lat, color='black', linewidth=width, ls=style, 
                transform=ccrs.PlateCarree())

def plot_colorbar(ax, clev):
    """Plot the colorbar."""
    norm = mpl.colors.Normalize(vmin=clev[0], vmax=clev[-1])
    cbar = mpl.colorbar.ColorbarBase(ax, cmap=pplt.Colormap('ColdHot'), norm=norm, 
                                     orientation='horizontal', extend='both', ticks=clev[::2])
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label(label='°C', size=9)

def plot_time_series(axs, lfcs):
    """Plot time series."""
    time_plot = np.linspace(1950, 2014, 65)
    var_plot = [lfcs[:, 0].reshape(-1, 12).mean(axis=1), 
                lfcs[:, 1].reshape(-1, 12).mean(axis=1) * (-1)]
    titles = ['(d) Principal component 1', '(e) Principal Component 2']
    
    for i, (ax, y) in enumerate(zip(axs, var_plot)):
        ax.plot(time_plot, y, color='k', linewidth=1.5)
        
        ax.fill_between(time_plot, y, where=y>=0, interpolate=True, color='#BF4146', alpha=0.5)
        ax.fill_between(time_plot, y, where=y<=0, interpolate=True, color='#80A1C2', alpha=0.5)
        
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlim(1950, 2014)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0, color='k', ls='--', lw=0.3)
        ax.set_title(titles[i], pad=-2)
    
    axs[0].set_ylabel('°C', size=9)
    axs[1].axes.get_yaxis().set_ticklabels([])

def main():
    plot_model = 'cobe'
    map_pattern1, map_pattern2, lfcs = load_data(plot_model)
    
    fig, (ax1, ax2, ax3, ax4, ax_cb1) = create_figure()
    
    plot_maps([ax1, ax2], [map_pattern1, map_pattern2 * (-1)], nle_tos.lon, nle_tos.lat)
    
    clev = np.arange(-0.4, 0.45, 0.05)
    plot_colorbar(ax_cb1, clev)
    
    plot_time_series([ax3, ax4], lfcs)
    
    plt.savefig('Fig_lfca_cobe.png', dpi=750, bbox_inches="tight")

if __name__ == "__main__":
    main()