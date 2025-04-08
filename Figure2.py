#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate and plot mean spatial maps of SST trend discrepancies.
"""

__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
import os
import sys
import itertools
import warnings

import numpy as np
import xarray as xr
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import proplot as pplt  # Keep if used in plot_map_gridspec

# Custom modules
from help_func import calc_sig
from import_highresmip_data import ersst_tos
from plot_map import plot_map_gridspec

warnings.filterwarnings('ignore')

# Global plot settings
plt.rc('font', size=12)
pplt.rc.update(grid=False)

#============================================================
#Constants 
DATA_DIR = '/work/mh0033/m300952/HighRESMIP/data/'
START_YEAR = 1979
END_YEAR = 2014

COLD_TONGUE_LATS = [-5, -5, 5, 5, -5]; COLD_TONGUE_LONS = [-95, 180, 180, -95, -95]
SEP_WARM_BIAS_LATS = [-20, -20, 0, 0, -20]; SEP_WARM_BIAS_LONS = [-70, -95, -95, -70, -70]
EASTERN_ATLANTIC_LATS = [-20, -20, 5, 5, -20]; EASTERN_ATLANTIC_LONS1 = [-20, 359, 359, -20, -20]; \
EASTERN_ATLANTIC_LONS2 = [0, 10, 10, 0, 0]
SOUTHERN_OCEAN_LATS = [-75, -75, -45, -45, -75]; SOUTHERN_OCEAN_LONS = [359, 0, 0, 359, 359]

COLORBAR_LABEL_SIZE = 9
ROBINSON_CENTRAL_LON = 180
LINEWIDTH = 0.3
DPI = 750

#============================================================
def load_and_preprocess_data(data_dir, start_yr, end_yr):
    """
    Load and preprocess the datasets.

    Args:
        data_dir (str): The directory containing the data.
        start_yr (str): The start year for time selection.
        end_yr (str): The end year for time selection.

    Returns:
        tuple[xr.DataArray, xr.DataArray, xr.DataArray]: A tuple containing the loaded and preprocessed datasets.
    """
    # Load the base dataset for time coordinates
    ds1_path = os.path.join(data_dir, 'cold', 'dc', 'tos_MPI-ER1_1950-2014_g025_yr_dc.nc')
    try:
        ds1 = xr.open_dataset(ds1_path).tos
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {ds1_path} was not found.")

    def preprocess(ds):
        """Assign time coordinates based on ds1."""
        try:
            ds = ds.assign_coords(time=ds1.time)
        except AttributeError as e:
            print(f"Error assigning time coordinates: {e}")
            raise
        return ds

    # Load data
    mods_cold_path = os.path.join(data_dir, 'cold', 'dc', 'tos*.nc')
    mods_warm_path = os.path.join(data_dir, 'warm', 'dc', 'tos*.nc')

    try:
        mods_cold = xr.open_mfdataset(mods_cold_path, combine='nested', concat_dim='n', coords='minimal', preprocess=preprocess).tos[:,:,:,:,0]
        mods_warm = xr.open_mfdataset(mods_warm_path, combine='nested', concat_dim='n', coords='minimal', preprocess=preprocess).tos
    except FileNotFoundError as e:
        raise FileNotFoundError(f"One or more data files not found: {e}")
    except ValueError as e:
        raise ValueError(f"Error opening multi-file dataset: {e}")

    # Time selections
    cold_tos = mods_cold.sel(time=slice(start_yr, end_yr))
    warm_tos = mods_warm.sel(time=slice(start_yr, end_yr))

    return cold_tos, warm_tos, ds1

def calculate_trends(cold_tos, warm_tos, obs_tos):
    """
    Calculate trends for the datasets.

    Args:
        cold_tos (xr.DataArray): Cold cold tongue TOS dataset.
        warm_tos (xr.DataArray): Warm cold tongue TOS dataset.
        obs_tos (xr.DataArray): Observational TOS dataset.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the calculated trends.
    """
    # Calculate trends
    tcold_tos = np.apply_along_axis(mk_test, 1, np.nan_to_num(cold_tos)) * 10
    twarm_tos = np.apply_along_axis(mk_test, 1, np.nan_to_num(warm_tos)) * 10
    tobs_tos = np.apply_along_axis(mk_test, 0, np.nan_to_num(obs_tos)) * 10
    
    return tcold_tos, twarm_tos, tobs_tos

def calculate_ensemble_means_difference(tcold_tos, twarm_tos):
    """
    Calculate the ensemble means for the datasets.

    Args:
        tcold_tos (np.ndarray): Cold cold tongue TOS trend dataset.
        twarm_tos (np.ndarray): Warm cold tongue TOS trend dataset.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the ensemble means of the datasets.
    """
    tcold_tos_em = np.nanmean(tcold_tos, axis=0)
    twarm_tos_em = np.nanmean(twarm_tos, axis=0)
    # Replace zero with NaN
    tcold_tos_em[tcold_tos_em == 0.0] = np.nan
    twarm_tos_em[twarm_tos_em == 0.0] = np.nan
    
    tos_diff = twarm_tos_em - tcold_tos_em

    return tcold_tos_em, twarm_tos_em, tos_diff

def perform_mann_whitney_u_test(tcold_tos, twarm_tos):
    """
    Perform the Mann-Whitney U test for the datasets.

    Args:
        tcold_tos (np.ndarray): Cold cold tongue TOS trend dataset.
        twarm_tos (np.ndarray): Warm cold tongue TOS trend dataset.

    Returns:
        ptos_diff (np.ndarray): Results of the Mann-Whitney U test.
    """
    ptos_diff = np.empty(twarm_tos[0].shape)
    for i in range(ptos_diff.shape[0]):
        for j in range(ptos_diff.shape[1]):
            ptos_diff[i, j] = mannwhitneyu(tcold_tos[:, i, j], twarm_tos[:, i, j])[1]
    ptos_diff[ptos_diff > 0.05] = np.nan
    ptos_diff[ptos_diff <= 0.05] = 1
    
    return ptos_diff

def calculate_p_values(cold_tos_m, warm_tos_m):
    """
    Calculate p-values for significance testing.

    Args:
        cold_tos_m (np.ndarray): Cold cold tongue TOS average trend dataset.
        warm_tos_m (np.ndarray): Warm cold tongue TOS average trend dataset.

    Returns:
        p_tos (np.ndarray): Calculated p-values.
    """
    p_tos = np.empty(warm_tos_m[0].shape)
    for i in range(p_tos.shape[0]):
        for j in range(p_tos.shape[1]):
            if np.unique(np.isnan(warm_tos_m[:, i, j] - cold_tos_m[:, i, j]))[0]:
                p_tos[i, j] = np.nan
            else:
                p_tos[i, j] = calc_sig(warm_tos_m[:, i, j] - cold_tos_m[:, i, j], a=0.05)
                
    return p_tos

def create_figure(lon, lat, tcold_tos_em, twarm_tos_em, tos_diff, ptos_diff):
    """
    Create the figure and save it.

    Args:
        lon (np.ndarray): Longitude data.
        lat (np.ndarray): Latitude data.
        tcold_tos_em (np.ndarray): Cold cold tongue TOS average trend data.
        twarm_tos_em (np.ndarray): Warm cold tongue TOS average trend data.
        tos_diff (np.ndarray): Difference data.
        ptos_diff (np.ndarray): P-value difference data.
    """

    fig = plt.figure(figsize=(3.39, 7.48))
    spec = gridspec.GridSpec(ncols=1, nrows=6,
                              height_ratios=[1, 1, 0.05, 0.1, 1, 0.05],
                              hspace=0.4, wspace=0.)

    projection = ccrs.Robinson(ROBINSON_CENTRAL_LON)
    ax1 = fig.add_subplot(spec[0, 0], projection=projection)
    ax2 = fig.add_subplot(spec[1, 0], projection=projection)
    ax3 = fig.add_subplot(spec[4, 0], projection=projection)
    ax_cb1 = fig.add_subplot(spec[2, :])
    ax_cb2 = fig.add_subplot(spec[5, :])

    axs = [ax1, ax2, ax3]
    titles = ['(a) Cold tongue bias < 0 | n = 97', '(b) Cold tongue bias ≥ 0 | n = 17', '(c) Difference']
    var = [tcold_tos_em, twarm_tos_em, tos_diff]
    levels = [np.arange(-0.4, 0.45, 0.05), np.arange(-0.4, 0.45, 0.05), np.arange(-0.2, 0.22, 0.02)]
    cmap = pplt.Colormap('ColdHot')

    for i, ax in enumerate(axs):
        plot_map_gridspec(ax, var[i], lon, lat, levels=levels[i], mp=0,  # Removed unused mp
                           cmap=cmap, ticks=False, land=True, title=titles[i],
                           loc_title='center', pad=5, fontsize=10)

        # Define regions of interest (use the constants)
        regions = [
            (SOUTHERN_OCEAN_LONS, SOUTHERN_OCEAN_LATS, 'ls'),
            (COLD_TONGUE_LONS, COLD_TONGUE_LATS, None),
            ([110, 180, 180, 110, 110], [-5, -5, 5, 5, -5], None),  # WP Lon/Lat
            (SEP_WARM_BIAS_LONS, [-45, -45, -5, -5, -45], 'ls'),
            ([-145, -110, -110, -145, -145], [10, 10, 30, 30, 10], 'ls')  # NE Pacific
        ]

        for lons, lats, linestyle in regions:
            ax.plot([l % 360 for l in lons], lats, color='black', linewidth=LINEWIDTH,
                    ls='--' if linestyle == 'ls' else '-', marker='',
                    transform=ccrs.PlateCarree())

    plt.rcParams['hatch.linewidth'] = 0
    plt.rcParams['hatch.color'] = '#3f3f3f'

    ax3.contourf(lon, lat, ptos_diff, 1, hatches=[4 * '.'],
                 alpha=0., transform=ccrs.PlateCarree())

    # Colorbars
    create_colorbar(ax_cb1, cmap, levels[0], label='$^\circ$C $decade^{-1}$')
    create_colorbar(ax_cb2, cmap, levels[2], label='$^\circ$C $decade^{-1}$')

    plt.suptitle('SST trend in models (1979 - 2014)', fontsize=11, y=0.95)
    plt.savefig('Fig2.png', dpi=DPI, bbox_inches="tight")
        
def main():
    """Main function for plotting."""
    data_dir = str(DATA_DIR)
    start_yr = str(START_YEAR)
    end_yr = str(END_YEAR)

    # Load and preprocess data
    cold_tos, warm_tos, ds1 = load_and_preprocess_data(data_dir, start_yr, end_yr)

    # Calculate observational means
    years = ds1.time.dt.year
    obs_tos = ersst_tos.sel(time=slice(start_yr, end_yr)).mean(axis=0)

    # Calculate trends
    tcold_tos, twarm_tos, tobs_tos = calculate_trends(cold_tos, warm_tos, obs_tos)

    # Calculate ensemble means and difference
    tcold_tos_em, twarm_tos_em, tos_diff = calculate_ensemble_means_difference(tcold_tos, twarm_tos)

    # Perform Mann-Whitney U test
    ptos_diff = perform_mann_whitney_u_test(tcold_tos, twarm_tos)

    # Calculate p-values
    cold_tos_m = np.nanmean(cold_tos, axis=0)
    warm_tos_m = np.nanmean(warm_tos, axis=0)
    p_tos = calculate_p_values(cold_tos_m, warm_tos_m)

    # Extract lat/lon from one of the datasets
    lon = ds1.lon.values
    lat = ds1.lat.values

    # Create the figure
    create_figure(lon, lat, tcold_tos_em, twarm_tos_em, tos_diff, ptos_diff)

if __name__ == "__main__":
    main()