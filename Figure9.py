  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trends over multiple time periods
"""

__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
## Load modules
import numpy as np
import pandas as pd
import seaborn as sns
import proplot as pplt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn import preprocessing

# Local modules
from import_highresmip_data import *
from help_func import *

# Configure global settings
pplt.rc.update({
    'grid': False,
    'font.size': 10,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False
})

#============================================================
# CONSTANTS 
START_YEAR = 1950
END_YEAR = 2014
MIN_TREND_YEARS = 30
VAR_NAMES = ['tos', 'tauu', 'theta']
DPI = 750

REGION_BOUNDS = {
    'lat': (-5, 5),
    'lon': (180%360, -95%360),
    'time': ('1950', '2014')
}

#============================================================
def load_and_preprocess_data(data, time, lon, lat, var_name):
    """Load and preprocess specified variable data"""
    lat1, lat2 = REGION_BOUNDS['lat']
    lon1, lon2 = REGION_BOUNDS['lon']
    
    if var_name == 'theta':
        data_ep, lons, lats = selreg(data.squeeze(), lon, lat, lat1=lat2, lat2=lat1, lon1=lon1, lon2=lon2)
        depth = data.depth
        d20 = np.apply_along_axis(calc_d20, -3, data_ep, depth=depth)
        d20m = np.stack([wgt_mean(v, lons, lats) for v in d20[0]])
        d20t = np.stack([calculate_thermocline_tilt(v) for v in d20[0]])
        return d20m, d20t, time
    
    else:
        data_ep = data.sel(
            lat=slice(lat1, lat2),
            lon=slice(lon1, lon2),
            time=slice(*REGION_BOUNDS['time'])
        ).mean(['lat', 'lon'])[0]
        return data_ep, None, None

def create_dataframe(trends, time):
    """Reshape 1D trend array to 2D DataFrame"""
    years = np.unique(time.dt.year)
    shape = int(np.sqrt(len(trends)))
    trends_reshaped = trends.reshape(shape, shape).T
    df = pd.DataFrame(
        trends_reshaped,
        columns=[str(v) for v in years]
    )
    df.replace(0, np.nan, inplace=True)
    df = df.set_index(np.arange(START_YEAR, END_YEAR-1))
    return df

def plot_heatmap(ax, data, vmin, vmax, title, cbar_label):
    """Create standardized heatmap plot"""
    cax = inset_axes(ax,
                     width="80%",
                     height="4%",
                     loc='upper right',
                     bbox_to_anchor=(0, -0.05, 1, 1),
                     bbox_transform=ax.transAxes,
                     borderpad=0)
    sns.heatmap(data, vmin=vmin, vmax=vmax, cmap=pplt.Colormap('NegPos'), center=0,
                xticklabels=4, yticklabels=4, square=True, linewidths=.002,
                ax=ax, cbar_ax=cax, cbar_kws={'orientation': 'horizontal'})
    cax.set_xlabel(cbar_label, fontsize=8)
    cax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=10)
    h = ax.collections[0]
    h.set_clim(vmin, vmax)

    ax.set_xlim(0, END_YEAR - START_YEAR - MIN_TREND_YEARS)
    ax.set_ylim(MIN_TREND_YEARS, END_YEAR - START_YEAR)
    ax.invert_yaxis()

def main():
    """Main analysis workflow"""
    # Load and process data
    sst_ep, _, _ = load_and_preprocess_data(ersst_tos, time, lon, lat, 'tos')
    tauu_ep, _, _ = load_and_preprocess_data(obs_tauu, time, lon, lat, 'tauu')
    d20m, d20t, time = load_and_preprocess_data(obs_theta, time, lon, lat, 'theta')
    
    # Calculate trends
    sst_trends = calculate_trends(sst_ep, time, str(START_YEAR), str(END_YEAR))
    tauu_trends = calculate_trends(tauu_ep, time, str(START_YEAR), str(END_YEAR))
    d20m_trends = calculate_trends(d20m, time, str(START_YEAR), str(END_YEAR))
    d20t_trends = calculate_trends(d20t, time, str(START_YEAR), str(END_YEAR))
    
    # Create DataFrames
    sst_df = create_dataframe(sst_trends, time)
    tauu_df = create_dataframe(tauu_trends, time)
    d20m_df = create_dataframe(d20m_trends, time)
    d20t_df = create_dataframe(d20t_trends, time)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 8.28))
    plt.subplots_adjust(hspace=0.08, wspace=0.14)
    axes = axes.flatten()

    # Plot heatmaps
    plot_heatmap(axes[0], sst_df, -0.4, 0.4, '(a) SST trend', '$^\circ$C decade$^{-1}$')
    plot_heatmap(axes[1], tauu_df, -0.005, 0.005, '(b) Zonal windstress trend', 'N $m^{-2}$ decade$^{-1}$')
    plot_heatmap(axes[2], d20m_df, -4, 4, '(c) Depth of 20$^\circ$C thermocline', 'm decade$^{-1}$')
    plot_heatmap(axes[3], d20t_df*(-1), -0.3, 0.3, '(d) Tilt of 20$^\circ$C thermocline', '$^\circ$ decade$^{-1}$')
    
    # Set labels
    axes[2].set_xlabel('Start year', fontsize=8)
    axes[0].set_ylabel('End year', fontsize=8)
    axes[2].set_ylabel('End year', fontsize=8)
    axes[3].set_xlabel('Start year', fontsize=8)

    plt.savefig('Fig9.png', dpi=DPI, facecolor='white', bbox_inches="tight")

#============================================================
## Execute script 
if __name__ == '__main__':
    main()