  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synchrony of trends: Kernel density estimate
"""

__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
## Load modules
import numpy as np
import seaborn as sns
import proplot as pplt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# Local modules
from import_highresmip_data import *
from help_func import *
from Figure9 import load_and_preprocess_data

# Configure global settings
pplt.rc.update({
    'grid': False,
    'font.size': 10,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False
})

#============================================================
# CONSTANTS 
REGION_BOUNDS = {
    'lat': (-5, 5),
    'lon': (180%360, -95%360),
    'time': ('1950', '2014')
}
DPI = 750
START_YEAR = 1950
END_YEAR = 2015
MIN_TREND_YEARS = 30
VAR_NAMES = ['tos', 'tauu', 'theta']
COLORS = ['#BF4A4A', '#6B8E23', '#3B7A8A', 'k']
YINC = [0, 0.1, 0.2, 0.3]
YLIM = (-0.002, 0.0025)
XLIM = (-0.2, 0.3)

#============================================================
def create_scatter_plot(ax, x_data, y_data, color, yinc, x_label, y_label, title):
    """Create standardized scatter plot"""
    ax.set_title(title, pad=-2, fontsize=12)
    ax.scatter(x_data, y_data, s=2, alpha=0.3, color=color, edgecolor='none')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Calculate and display correlation coefficient
    r, _ = stats.pearsonr(np.nan_to_num(x_data), np.nan_to_num(y_data))
    ax.text(0.25, 0.65 + yinc, f'{r:.2f}', transform=ax.transAxes, color=color,
            verticalalignment='top', ha='right', fontsize=12)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    # Add KDE plots
    sns.kdeplot(ax=ax, x=x_data, y=y_data, shade=False, color=color,
                linewidth=0.5, bw_adjust=1.2, levels=1)

def main():
    # Combine data
    data_sst = [ll_tos, mm_tos, hm_tos, ersst_tos]
    data_tauu = [ll_tauu, mm_tauu, hm_tauu, obs_tauu]
    data_theta = [ll_theta, mm_theta, hm_theta, obs_theta]

    # Load and preprocess data
    data_sst_ep = [load_and_preprocess_data(v, lon, lat, 'tos') for v in data_sst]
    data_tauu_ep = [load_and_preprocess_data(v, lon, lat, 'tauu') for v in data_tauu]
    data_theta_ep = [load_and_preprocess_data(v, lon, lat, 'theta') for v in data_theta]

    # Calculate depth of 20C thermocline
    depth = []
    for i in range(len(data_theta_ep)):
        try:
            depth.append(data_theta_ep[i].lev)
        except AttributeError:
            depth.append(data_theta_ep[i].depth)

    d20 = [np.apply_along_axis(calc_d20, -3, data_theta_ep[i], depth=depth[i]) for i in range(len(depth))]

    # Calculate tilt
    d20t = [np.stack([calculate_thermocline_tilt(v) for v in d20[i]]) for i in range(len(d20))]

    # Calculate trends
    trends_sst = [calculate_trends(v, time) for v in data_sst_ep]
    trends_tauu = [calculate_trends(v, time) for v in data_tauu_ep]
    trends_d20t = [calculate_trends(v, time) for v in d20t]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 3))
    plt.subplots_adjust(wspace=0.35)
    ax1, ax2, ax3 = axes

    # Create scatter plots
    create_scatter_plot(ax1, trends_sst[-1], trends_tauu[-1], COLORS[-1], YINC[-1],
                        'SST [$^\circ$C $decade^{-1}$]', 'Surface zonal windstress\n[N $m^{-1}$ $decade^{-1}$]', '(a)')
    create_scatter_plot(ax2, trends_d20t[-1], trends_sst[-1], COLORS[-1], YINC[-1],
                        'Thermocline tilt [$^\circ$ $decade^{-1}$]', 'SST [$^\circ$C $decade^{-1}$]', '(b)')
    create_scatter_plot(ax3, trends_tauu[-1], trends_d20t[-1], COLORS[-1], YINC[-1],
                        'Surface zonal windstress\n[N $m^{-1}$ $decade^{-1}$]',
                        'Thermocline tilt [$^\circ$ $decade^{-1}$]', '(c)')

    # Adjust axis limits
    ax1.set_ylim(*YLIM)
    ax3.set_xlim(*YLIM)
    ax1.set_xlim(*XLIM)
    ax2.set_ylim(*XLIM)
    ax3.set_ylim(-0.0005, 0.0008)
    ax2.set_xlim(-0.0005, 0.0008)

    # Remove top and right spines
    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Create custom legend handles
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='none') for color in COLORS]

    # Add legend
    ax3.legend(legend_elements, ["LL", "MM", "HM", 'Observed'], bbox_to_anchor=(0.56, 0.4), frameon=False)

    plt.suptitle('Synchrony of trends in HadGEM3-GC31 models', y=1, fontsize=11)
    plt.tight_layout()
    plt.savefig('Fig10.png', dpi=DPI, bbox_inches="tight")

#============================================================
## Execute script 
if __name__ == '__main__':
    main()