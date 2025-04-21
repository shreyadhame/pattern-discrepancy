  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transient Ocean Dynamical Thermostat (1979 - 2014) in HadGEM3-GC31 models
"""

__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
## Load modules
import os
import warnings
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Local modules
from import_highresmip_data import *
from help_func import *
from plot_map import *

# Matplotlib and Proplot configuration
plt.rc('font', size=10)
pplt.rc.update(grid=False)

#============================================================
# CONSTANTS 
START_YEAR = 1979
END_YEAR = 2014
DPI = 750
DATA_PATH = '/work/mh0033/m300952/HighRESMIP/'
LAT_RANGE_EP = (-5, 5)
LON_RANGE_EP = (180%360, -95%360)
LAT_RANGE_ZONAL = (-20, 20)
LEVEL_RANGE = (0, 250)

#============================================================
def load_data(path, model, resolution):
    """
    Load WO data for a given model and resolution.
    """
    filename = f'wo_PrimOmon_HadGEM3-GC31-{resolution}_hist-1950_r1i1p1f1_0-500_1950-2014_g025_yr.nc'
    full_path = os.path.join(path, 'MOHC', model, 'wo', filename)
    try:
        ds = xr.open_dataset(full_path)
        return ds.wo
    except FileNotFoundError:
        print(f"Error: File not found at {full_path}")
        return None

def select_time_range(data, start_yr, end_yr):
    """
    Select a time range from the data.
    """
    return data.sel(time=slice(start_yr, end_yr))

def select_region(data, lat_range, lon_range, level_range=None):
    """
    Select a region from the data based on latitude, longitude, and optional level.
    """
    subset = data.sel(lat=slice(*lat_range), lon=slice(*lon_range))
    if level_range is not None:
        subset = subset.sel(lev=slice(*level_range))
    return subset

def interpolate_data(data, target_levels, axis=1):
    """
    Interpolate data to a new set of levels using linear interpolation.
    """
    finterp = interp1d(data.lev, data, axis=axis, fill_value="extrapolate")
    return finterp(target_levels)

def plot_data(lons, lats, depths,
              tthetat_epm, tthetat_epz, ttauut_ep,
              pthetat_epm, pthetat_epz, ptauut_ep,
              wdT_dzm, wdT_dzz, tauut_epm, filename):
    """
    Plot the trends and significance masks for all datasets.
    """
    fig = plt.figure(figsize=(3.74, 9.7))
    widths = [1, 0.5]
    heights = [0.5, 0.05, 1, 0.1, 0.5, 0.05, 1, 0.1, 0.5, 0.05, 1, 0.05]
    spec = gridspec.GridSpec(ncols=2, nrows=len(heights), width_ratios=widths,
                              height_ratios=heights, hspace=0.5, wspace=0.1)

    ax1 = fig.add_subplot(spec[0, 0])
    g1 = fig.add_subplot(spec[1, 0])
    ax2 = fig.add_subplot(spec[2, 0])
    ax3 = fig.add_subplot(spec[2, 1])
    g2 = fig.add_subplot(spec[3, 0])

    ax4 = fig.add_subplot(spec[4, 0])
    g3 = fig.add_subplot(spec[5, 0])
    ax5 = fig.add_subplot(spec[6, 0])
    ax6 = fig.add_subplot(spec[6, 1])
    g4 = fig.add_subplot(spec[7, 0])

    ax7 = fig.add_subplot(spec[8, 0])
    g5 = fig.add_subplot(spec[9, 0])
    ax8 = fig.add_subplot(spec[10, 0])
    ax9 = fig.add_subplot(spec[10, 1])

    cb = fig.add_subplot(spec[11, :])

    var = [tthetat_epm[0], tthetat_epz[0], tthetat_epm[1], tthetat_epz[1], tthetat_epm[2], tthetat_epz[2]]
    var_tauu = [ttauut_ep[0], ttauut_ep[1], ttauut_ep[2]]
    cont = [wdT_dzm[0], wdT_dzz[0], wdT_dzm[1], wdT_dzz[1], wdT_dzm[2], wdT_dzz[2]]
    pvar = [pthetat_epm[0], pthetat_epz[0], pthetat_epm[1], pthetat_epz[1], pthetat_epm[2], pthetat_epz[2]]
    pvar_tauu = [ptauut_ep[0], ptauut_ep[1], ptauut_ep[2]]
    cont_tauu = [tauut_epm[0], tauut_epm[1], tauut_epm[2]]
    axs_theta = [ax2, ax3, ax5, ax6, ax8, ax9]
    axs_tauu = [ax1, ax4, ax7]
    x = [lons, lats, lons, lats, lons, lats]
    titles = ['LL', 'MM', 'HM']

    cmap = pplt.Colormap('ColdHot')

    # Plot theta data
    levels_theta = np.arange(-0.6, 0.65, 0.05)
    for i, ax in enumerate(axs_theta):
        cf = ax.contourf(x[i], depths, var[i], levels=levels_theta, cmap=cmap, extend='both')
        contours = ax.contour(x[i], depths, cont[i] * 1e7, levels=np.round(np.linspace(-8, 8, 8), 1), colors='k', linewidths=0.5)
        ax.clabel(contours, inline=True, fontsize=6)
        ax.invert_yaxis()
        ax.contourf(x[i], depths, pvar[i], 1, hatches=[4 * '.'], alpha=0.)

    # Plot tauu data
    levels_tauu = np.arange(-0.006, 0.0065, 0.0005)
    for i, ax in enumerate(axs_tauu):
        cf = ax.contourf(lons, lats, var_tauu[i], levels=levels_tauu, cmap=cmap, extend='both')
        ax.axhline(0, linestyle='--', color='k', lw=0.5, alpha=0.8)
        ax.contourf(lons, lats, pvar_tauu[i], 1, hatches=[4 * '.'], alpha=0.)

    # Customize axes
    for axs in [ax1, ax4, ax7]:
        axs.set_title('Surface zonal wind stress trend [$10^{-2}$ N $m^{-2}$ $decade^{-1}$]', pad=-2, loc='left', fontsize=10)
        axs.get_xaxis().set_ticklabels([])
        axs.get_yaxis().set_ticklabels(['20$^\circ$S', '10$^\circ$S', '0$^\circ$', '10$^\circ$N', '20$^\circ$N'], fontsize=9)

    for axs in [ax2, ax5, ax8]:
        axs.set_title('Meridional mean', pad=-2, loc='left', fontsize=10)
        axs.set_ylabel('Depth [m]', fontsize=9)
        axs.set_yticks([50, 100, 150, 200])
        axs.set_yticklabels([50, 100, 150, 200], fontsize=9)
        axs.get_xaxis().set_ticklabels(['180$^\circ$', '160$^\circ$W', '135$^\circ$W', '110$^\circ$W'], fontsize=9)

    for axs in [ax3, ax6, ax9]:
        axs.set_yticks([50, 100, 150, 200])
        axs.set_title('Zonal mean', pad=-2, loc='left', fontsize=10)
        axs.axvline(0, linestyle='--', color='k', lw=0.5, alpha=0.8)
        axs.get_yaxis().set_ticklabels([])
        axs.get_xaxis().set_ticklabels(['20$^\circ$S', '10$^\circ$S', '0$^\circ$', '10$^\circ$N', '20$^\circ$N'], fontsize=9)

    # Add labels
    for g, text in zip([g1, g3, g5], ['(a) HadGEM3-GC31-LL | n=8', '(b) HadGEM3-GC31-MM | n=3', '(c) HadGEM3-GC31-HM | n=3']):
        g.axis('off')
        g.text(0., 0.2, 'Subsurface temperature trend [$^\circ$C $decade^{-1}$]', fontsize=10)
        g.text(0, 19.5, text, fontsize=12)

    for g in [g2, g4]:
        g.axis('off')

    for axs in [ax2, ax3, ax5, ax6]:
        axs.set_xticklabels([])

    # Colorbar
    vmin = levels_theta[0]
    vmax = np.round(levels_theta[-1], 2)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = mpl.colorbar.ColorbarBase(cb, cmap=cmap, norm=norm, orientation='horizontal',
                                     extend='both', ticks=levels_theta[::4])
    cbar.ax.tick_params(labelsize=9)

    plt.suptitle('Simulated wind and thermocline trends', fontsize=11, y=0.94)
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches="tight")

def main():
    # Load data
    ll_wo = load_data(DATA_PATH, 'MOHC', 'LL')
    mm_wo = load_data(DATA_PATH, 'MOHC', 'MM')
    hm_wo = load_data(DATA_PATH, 'MOHC', 'HM')
    
    ll_theta = load_data(DATA_PATH, 'MOHC', 'LL')
    mm_theta = load_data(DATA_PATH, 'MOHC', 'MM')
    hm_theta = load_data(DATA_PATH, 'MOHC', 'HM')

    ll_tauu = load_data(DATA_PATH, 'MOHC', 'LL')
    mm_tauu = load_data(DATA_PATH, 'MOHC', 'MM')
    hm_tauu = load_data(DATA_PATH, 'MOHC', 'HM')

    data_theta = [ll_theta, mm_theta, hm_theta]
    data_wo = [ll_wo, mm_wo, hm_wo]
    data_tauu = [ll_tauu, mm_tauu, hm_tauu]

    # Select time range
    thetat = [select_time_range(v, str(START_YEAR), str(END_YEAR)) for v in data_theta]
    wt = [select_time_range(v, str(START_YEAR), str(END_YEAR)) for v in data_wo]
    tauut = [select_time_range(v, str(START_YEAR), str(END_YEAR)) for v in data_tauu]

    # Common coordinates
    lon = ll_theta.lon
    lat = ll_theta.lat
    lons = lon.sel(lon=slice(LON_RANGE_EP[0], LON_RANGE_EP[1]))
    lats = lat.sel(lat=slice(LAT_RANGE_ZONAL[0], LAT_RANGE_ZONAL[1]))
    depths = ll_theta.lev.sel(lev=slice(*LEVEL_RANGE))

    # Meridional mean (EP) analysis
    thetat_epm = [
        select_region(v, LAT_RANGE_EP, LON_RANGE_EP, LEVEL_RANGE).mean(('lat', 'n'))
        for v in thetat
    ]
    tthetat_epm = [compute_trend_and_significance(v, 0)[0] for v in thetat_epm]
    pthetat_epm = [compute_trend_and_significance(v, 0)[1] for v in thetat_epm]

    tauut_ep = [
        select_region(v, (-20, 20), LON_RANGE_EP).mean('n')
        for v in tauut
    ]
    ttauut_ep = [compute_trend_and_significance(v, 0)[0] for v in tauut_ep]
    ptauut_ep = [compute_trend_and_significance(v, 0)[1] for v in tauut_ep]
    tauut_epm = [np.nanmean(v, axis=0) for v in tauut_ep]

    wt_epm = [
        select_region(v, LAT_RANGE_EP, LON_RANGE_EP).mean('lat')
        for v in wt
    ]
    wti_epm = [interpolate_data(v, depths, axis=1) for v in wt_epm]
    dT_dz = [np.gradient(v, depths, axis=1) for v in thetat_epm]
    wdT_dzm = [np.nanmean((wti_epm[i] * dT_dz[i]), axis=0) for i in range(len(dT_dz))]

    # --- Zonal mean (EP) analysis
    thetat_epz = [
        select_region(v, LAT_RANGE_ZONAL, LON_RANGE_EP, LEVEL_RANGE).mean(('lon', 'n'))
        for v in thetat
    ]
    tthetat_epz = [compute_trend_and_significance(v, 0)[0] for v in thetat_epz]
    pthetat_epz = [compute_trend_and_significance(v, 0)[1] for v in thetat_epz]

    wt_epz = [
        select_region(v, LAT_RANGE_ZONAL, LON_RANGE_EP).mean('lon')
        for v in wt
    ]
    wti_epz = [interpolate_data(v, depths, axis=1) for v in wt_epz]
    dT_dz = [np.gradient(v, depths, axis=1) for v in thetat_epz]
    wdT_dzz = [np.nanmean((wti_epz[i] * dT_dz[i]), axis=0) for i in range(len(dT_dz))]

    # --- Plotting
    plot_data(
        lons, lats, depths,
        tthetat_epm, tthetat_epz, ttauut_ep,
        pthetat_epm, pthetat_epz, ptauut_ep,
        wdT_dzm, wdT_dzz, tauut_epm,
        filename = 'Fig8.png'
    )

# ============================================================
## Execute script
if __name__ == '__main__':
    main()