  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transient Ocean Dynamical Thermostat (1979 - 2014)
"""

__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
## Load modules
import warnings
import numpy as np

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

#============================================================
def mask_significance(pval_array, threshold=0.05):
    """
    Mask p-values above the significance threshold.
    """
    mask = np.where(pval_array > threshold, np.nan, 1)
    return mask

def select_region(data, lat_range, lon_range, depth_range=None):
    """
    Select a region from the data based on latitude, longitude, and optional depth.
    """
    subset = data.sel(lat=slice(*lat_range), lon=slice(*lon_range))
    if depth_range is not None:
        subset = subset.sel(depth=slice(*depth_range))
    return subset

def plot_results(lons, lats20, depths, latz, tvar, pvar, filename):
    """
    Plot the trends and significance masks.
    """
    fig = plt.figure(figsize=(4.23, 4.))
    widths = [1, 0.5]
    heights = [0.5, 0.04, 1, 0.04]
    spec = gridspec.GridSpec(
        ncols=2, nrows=4, width_ratios=widths,
        height_ratios=heights, hspace=0.4, wspace=0.1
    )

    ax1 = fig.add_subplot(spec[0, 0])
    gap = fig.add_subplot(spec[1, 0])
    gap.axis('off')
    ax2 = fig.add_subplot(spec[2, 0])
    ax3 = fig.add_subplot(spec[2, 1])
    cb = fig.add_subplot(spec[3, :])

    levels = [
        np.arange(-0.006, 0.0065, 0.0005),
        np.arange(-0.6, 0.65, 0.05),
        np.arange(-0.6, 0.65, 0.05)
    ]
    axs = [ax1, ax2, ax3]
    x = [lons, lons, latz]
    y = [lats20, depths, depths]
    cmap = pplt.Colormap('ColdHot')

    for i, ax in enumerate(axs):
        cf = ax.contourf(x[i], y[i], tvar[i], levels=levels[i], cmap=cmap, extend='both')
        plt.rcParams['hatch.linewidth'] = 0
        plt.rcParams['hatch.color'] = '#3f3f3f'
        ax.contourf(x[i], y[i], pvar[i], 1, hatches=[4 * '.'], alpha=0.)

    # Axes formatting
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax1.get_xaxis().set_ticklabels([])
    ax3.get_yaxis().set_ticklabels([])

    ax1.get_yaxis().set_ticklabels(['20$^\circ$S', '10$^\circ$S', '0$^\circ$', '10$^\circ$N', '20$^\circ$N'], fontsize=9)
    ax2.get_xaxis().set_ticklabels(['180$^\circ$', '160$^\circ$W', '140$^\circ$W', '120$^\circ$W', '100$^\circ$W'], fontsize=9)
    ax2.get_yaxis().set_ticklabels([0, 50, 100, 150, 200], fontsize=9)

    ax2.set_ylabel('Depth [m]', fontsize=9)
    ax3.axvline(0, linestyle='--', color='k', lw=0.5, alpha=0.8)
    ax1.axhline(0, linestyle='--', color='k', lw=0.5, alpha=0.8)
    ax1.axhline(-5, linestyle='-', color='k', lw=0.5, alpha=0.8)
    ax1.axhline(5, linestyle='-', color='k', lw=0.5, alpha=0.8)

    # Colorbar
    vmin = levels[-1][0]
    vmax = np.round(levels[-1][-1], 2)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = mpl.colorbar.ColorbarBase(
        cb, cmap=cmap, norm=norm, orientation='horizontal',
        extend='both', ticks=levels[-1][::4]
    )
    cbar.ax.tick_params(labelsize=10)

    # Titles
    ax1.set_title('Surface zonal wind stress [$10^{-2}$ N $m^{-2}$ $decade^{-1}$]', pad=-2, loc='left', fontsize=11)
    ax2.set_title('Meridional mean', pad=-2, loc='left', fontsize=11)
    ax3.set_title('Zonal mean', pad=-2, loc='left', fontsize=11)
    gap.text(0., 0.2, 'Subsurface temperature [$^\circ$C $decade^{-1}$]', fontsize=11)
    plt.suptitle('(a) Observed wind and thermocline trends', fontsize=12, fontweight='normal')

    plt.savefig(filename, dpi=DPI, bbox_inches="tight")

def main():
    # Time period selection
    obs_tauut = obs_tauu.sel(time=slice(str(START_YEAR), str(END_YEAR)))
    obs_thetat = obs_theta.sel(time=slice(str(START_YEAR), str(END_YEAR)))

    # Surface wind stress, meridional mean (EP)
    lat1, lat2 = -20, 20
    lon1, lon2 = 180 % 360, -95 % 360

    lons = lon.sel(lon=slice(lon1, lon2))
    lats20 = lat.sel(lat=slice(lat1, lat2))

    obs_tauu_ep = obs_tauut.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).mean('n')
    t_tauu_ep, p_tauu_ep = compute_trend_and_significance(obs_tauu_ep, 0)

    # Subsurface temperature, meridional mean (EP)
    lat1, lat2 = -5, 5
    depth_range = (0, 250)
    depths = obs_thetat.depth.sel(depth=slice(*depth_range))

    obs_theta_epm = obs_thetat.sel(
        lat=slice(lat1, lat2), lon=slice(lon1, lon2), depth=slice(*depth_range)
    ).mean('lat').mean('n')
    t_theta_epm, p_theta_epm = compute_trend_and_significance(obs_theta_epm, 0)

    # Subsurface temperature, zonal mean (EP)
    lat1, lat2 = -20, 20
    latz = lat.sel(lat=slice(lat1, lat2))

    obs_theta_epz = obs_thetat.sel(
        lat=slice(lat1, lat2), lon=slice(lon1, lon2), depth=slice(*depth_range)
    ).mean('lon').mean('n')
    t_theta_epz, p_theta_epz = compute_trend_and_significance(obs_theta_epz, 0)

    # Plotting
    plot_results(
        lons, lats20, depths, latz,
        [t_tauu_ep, t_theta_epm, t_theta_epz],
        [p_tauu_ep, p_theta_epm, p_theta_epz],
        filename = 'Fig7_1.png'
    )

# ============================================================
## Execute script
if __name__ == '__main__':
    main()