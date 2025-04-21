  #!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
## Load modules
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

warnings.filterwarnings('ignore')

# Local modules
from import_highresmip_data import *
from help_func import *
from plot_map import *

#============================================================
# CONSTANTS 
START_YEAR = 1979
END_YEAR = 2014

MARKERS = ["o", "s", "X", "X", "o", "s", "X", "X"]
COLORS = ["#56B4E9", "#56B4E9", "#56B4E9", "#56B4E9",
          "#2f2f2f", "#2f2f2f", "#2f2f2f", "#2f2f2f"]
MODEL_NAMES = ['LL', 'MM', 'HM', 'HH', 'LR', 'HR', 'XR', 'ER']
DPI = 750

# Region Definitions
LAT_MIN = -5
LAT_MAX = 5
EP_LON_MIN = 180%360
EP_LON_MAX = -95%360
WP_LON_MIN = 110%360
WP_LON_MAX = 180%360

# Data and Model Labels
DATA_NAMES = ["LL", "MM", "HM", "LR", "HR", "ER", "ERSSTv5", "COBE"]
PALETTE = ["#56B4E9", "#56B4E9", "#56B4E9", "#2f2f2f", "#2f2f2f", "#2f2f2f", "#AA4499", "#AA4499"]
# Plotting settings
FLIERPROPS = dict(marker='o', markersize=1, markeredgecolor='#565656', markerfacecolor='none', alpha=0.5)

#============================================================
def calculate_regional_mean(data, lat_min, lat_max, lon_min, lon_max):
    """
    Calculate the regional mean of the data.
    """
    return data.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).mean(('lat', 'lon'))

def long_term_trends(var, time, start_year, end_year):
    """
    Calculate long-term trends for a given variable.
    """
    trends = []
    for i in range(start_year, end_year):
        for j in range(start_year, end_year):
            ind1 = np.where(time.dt.year == i)[0][0]  # Index of start year
            ind2 = np.where(time.dt.year == j)[0][0]  # Index of end year
            chunk = var[ind1:ind2]  # Select a chunk
            if 30 <= len(chunk) <= 65:
                trends.append(mk_test(chunk)[0] * 10)
    return trends

def create_dataframe(data, model_names):
    """
    Create a Pandas DataFrame from the data and model names.
    """
    data_list = [[v, name] for arr, name in zip(data, model_names) for v in arr]
    return pd.DataFrame(data_list, columns=[" ", "model"])

# ============================================================
def calculate_mean_windstress_thermocline_biases(data_tauu, data_theta):
    # Windstress Data Preparation
    data_tauu_mean = [v.mean(axis=0) for v in data_tauu]

    # Potential Temperature Data Preparation
    data_theta_mean = [v.mean(axis=0) for v in data_theta]

    # Select Time Range
    obs_tauut = obs_tauu.sel(time=slice(str(START_YEAR), str(END_YEAR)))
    obs_thetat = obs_theta.sel(time=slice(str(START_YEAR), str(END_YEAR)))

    mod_tauu = [v.sel(time=slice(str(START_YEAR), str(END_YEAR))) for v in data_tauu_mean]
    mod_theta = [v.sel(time=slice(str(START_YEAR), str(END_YEAR))) for v in data_theta_mean]

    # Regional Selection
    obs_tauu_ep, lons, lats = selreg(
        obs_tauut.squeeze(), lon, lat,
        lat1=LAT_MIN, lat2=LAT_MAX,
        lon1=EP_LON_MIN, lon2=EP_LON_MAX
    )
    obs_tauu_epm = wgt_mean(obs_tauu_ep[0], lons, lats)

    mod_tauu_ep = [
        selreg(v, lon, lat,
            lat1=LAT_MIN, lat2=LAT_MAX,
            lon1=EP_LON_MIN, lon2=EP_LON_MAX)[0]
        for v in mod_tauu
    ]
    mod_tauu_epm = np.stack([wgt_mean(v, lons, lats) for v in mod_tauu_ep])

    obs_theta_ep, lons, lats = selreg(
        obs_thetat.squeeze().mean(axis=0), lon, lat,
        lat1=LAT_MIN, lat2=LAT_MAX,
        lon1=EP_LON_MIN, lon2=EP_LON_MAX
    )
    mod_theta_ep = [
        selreg(v.mean(axis=0), lon, lat,
            lat1=LAT_MIN, lat2=LAT_MAX,
            lon1=EP_LON_MIN, lon2=EP_LON_MAX)[0]
        for v in mod_theta
    ]

    # Thermocline Depth Calculation
    obs_depth = obs_theta.depth

    mod_depth = [
        getattr(v, 'lev', getattr(v, 'depth', None))
        for v in data_theta
    ]

    obs_d20 = np.apply_along_axis(calc_d20, -3, obs_theta_ep, depth=obs_depth)
    obs_d20_epm = wgt_mean(obs_d20, lons, lats)

    mod_d20 = [
        np.apply_along_axis(calc_d20, -3,
                            mod_theta_ep[i], depth=mod_depth[i])
        for i in range(len(mod_depth))
    ]
    mod_d20_epm = [wgt_mean(v, lons, lats) for v in mod_d20]

    # Bias Calculations
    # Windstress Bias
    ws_bias = mod_tauu_epm - obs_tauu_epm

    # Thermocline Depth Bias
    d20m_bias = [v - obs_d20_epm for v in mod_d20_epm]

    # Thermocline Tilt and Bias Calculation
    obs_d20_eps = calculate_thermocline_tilt(obs_d20)
    mod_d20_eps = [calculate_thermocline_tilt(v) for v in mod_d20]
    d20s_bias = [v - obs_d20_eps for v in mod_d20_eps]

    return ws_bias, d20m_bias, d20s_bias

def plot_mean_windstress_thermocline_bias(ws_bias, d20m_bias, d20s_bias, filename):
    """
    Plot mean equatorial zonal wind and thermocline structure bias.

    """
    try:
        fig = plt.figure(figsize=(6.5, 3.))
        widths = [1, 1]
        heights = [1]

        spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=widths,
                                height_ratios=heights, hspace=0., wspace=0.1)

        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[0, 1])

        # Plot wind stress bias
        x = np.arange(len(ws_bias))
        y = ws_bias
        for i in range(len(COLORS)):
            markerline, stemlines, baseline = ax1.stem(x[i], y[i], markerfmt=MARKERS[i])
            plt.setp(stemlines, 'color', COLORS[i])
            plt.setp(markerline, 'color', COLORS[i])
            plt.setp(markerline, 'markeredgecolor', COLORS[i])
            plt.setp(markerline, 'markeredgewidth', 1.5)
            plt.setp(markerline, 'markerfacecolor', 'none')
            plt.setp(markerline, 'markersize', 8)

            # Titles
            ax1.set_title('(a) Surface zonal windstress bias', fontsize=12, pad=-2)
            # Set limits
            ax1.set_ylim(-0.02, 0.02)
            # Zero lines
            ax1.axhline(0., color='#909090', zorder=-1, linewidth=1)

            # Add labels
            lines = [3.5]
            [ax1.axvline(x=i, linestyle='--', color='k', lw=0.5, alpha=0.8) for i in lines]
            y_text = -0.022
            inc = np.array(lines)
            ax1.text(inc[0] - 2, y_text, 'HadGEM3-GC31', fontsize=9, color=COLORS[2], ha='center', va='center')
            ax1.text(inc[0] + 3 - 1., y_text, 'MPI-ESM1-2', fontsize=9, color=COLORS[-1], ha='center', va='center')

            ax1.spines[['right', 'top']].set_visible(False)

        # Plot thermocline bias
        x = d20m_bias
        y = d20s_bias
        for i in range(len(COLORS)):
            ax2.scatter(x[i], y[i], s=50, marker=MARKERS[i], facecolors='none', edgecolors=COLORS[i], lw=1.5)

        # Fit a linear regression line
        b, a = np.polyfit(x, y, deg=1)
        xseq = np.linspace(np.nanmin(x), np.nanmax(x), num=100)
        # Plot the regression line
        ax2.plot(xseq, a + b * xseq, color="#909090", lw=0.4, ls='--')

        # Titles
        ax2.set_title('(b) Thermocline bias', fontsize=12, pad=-2)
        # Set limits
        ax2.set_ylim(-0.002, 0.012)

        ax2.spines[['left', 'top']].set_visible(False)

        # Zero lines
        ax2.axhline(0., color='#909090', zorder=-1, linewidth=1.)
        ax2.axvline(0., color='#909090', zorder=-1, linewidth=1.)

        # Labels
        ax1.set_xticks([])
        ax1.set_ylabel('N $m^{-2}$')
        ax2.set_xlabel('Bias in mean depth of thermocline [m]')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel('Bias in east-west thermocline tilt [$^\circ$]')

        # Add labels
        TEXTS = []
        for idx, model in enumerate(MODEL_NAMES):
            x_text, y_text = np.arange(len(ws_bias))[idx] - 0.3, ws_bias[idx]
            if y_text < 0:
                y_text = y_text - 0.004
            elif y_text > 0:
                y_text = y_text + 0.002
            TEXTS.append(ax1.text(x_text, y_text, model, fontsize=10, color=COLORS[idx]))

        TEXTS = []
        for idx, model in enumerate(MODEL_NAMES):
            x_text, y_text = d20m_bias[idx], d20s_bias[idx]
            TEXTS.append(ax2.text(x_text, y_text, model, fontsize=10, color=COLORS[idx]))

        # Adjust text position and add lines
        adjust_text(
            TEXTS,
            expand_points=(5, 5),
            arrowprops=dict(arrowstyle="-", lw=0.4),
            ax=ax2
        );

        # Legend
        circle = mlines.Line2D([], [], markerfacecolor='none', markeredgecolor='k', marker='o', linestyle='None',
                                markersize=8, label='Low resolution')
        cross = mlines.Line2D([], [], markerfacecolor='none', markeredgecolor='k', color='k', marker='X', linestyle='None',
                                markersize=8, label='Higher resolution')
        square = mlines.Line2D([], [], markerfacecolor='none', markeredgecolor='k', color='k', marker='s', linestyle='None',
                                markersize=8, label='Medium resolution')

        leg = ax2.legend(handles=[cross, square, circle], bbox_to_anchor=(0.88, 0.98), prop={'size': 10})
        leg.get_frame().set_linewidth(0.0)

        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        plt.suptitle('Mean equatorial zonal wind and thermocline structure', y=1.02, fontsize=11)

        plt.savefig(filename, dpi=DPI, bbox_inches="tight")
        print(f"Mean windstress and thermocline bias plot saved to {filename}")

    except Exception as e:
        print(f"Error in plot_mean_windstress_thermocline_bias: {e}")
    
def plot_longterm_trends(df_wp, df_ep, data_wp_tt, data_ep_tt, data_wp_ltr, data_ep_ltr, filename):
    """
    Plot the long-term trends in West Pacific and East Pacific regions.
    """
    try:
        # Create the Plot
        fig = plt.figure(figsize=(6.5, 6.5))
        widths = [1, 1]
        heights = [1, 1]

        # Set global settings to remove minor ticks
        pplt.rc.update({'ytick.minor.visible': False})

        spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=widths,
                                height_ratios=heights, hspace=0.2, wspace=0.08)

        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[0, 1])

        # Boxplots
        bx1 = sns.boxplot(ax=ax1, x=df_wp[' '], y=df_wp['model'], color='white', width=0.5,
                        flierprops=FLIERPROPS, medianprops=dict(color='#565656'))
        bx2 = sns.boxplot(ax=ax2, x=df_ep[' '], y=df_ep['model'], color='white', width=0.5,
                        flierprops=FLIERPROPS, medianprops=dict(color='#565656'))

        # Customize boxplot lines
        for bx in [bx1, bx2]:
            [v.set_linewidth(1) for v in bx.lines]

        # Limits and labels
        for ax in [ax1, ax2]:
            ax.set_xlim(-0.3, 0.4)
            lines = [2.5, 5.5, 7.5]
            [ax.axhline(y=i, linestyle='--', color='k', lw=0.5, alpha=0.8) for i in lines]
            ax.tick_params(axis='y', labelsize=11)  # Y-axis tick labels
            ax.axvline(0, color="#909090", lw=0.4, ls='--')

            for i, artist in enumerate(ax.artists):
                artist.set_edgecolor(PALETTE[i])
                artist.set_linewidth(1.)
                artist.set_facecolor('None')

        # Model labels
        colors_model = ["#56B4E9", "#2f2f2f", "#AA4499"]
        x = -0.45
        ax1.text(x, lines[1] - 4.5, 'HadGEM3\n-GC31', fontsize=9, rotation='vertical', color=colors_model[0],
                ha='center', va='center')
        ax1.text(x, lines[2] - 3.5, 'MPI-ESM1-2', fontsize=9, rotation='vertical', color=colors_model[1],
                ha='center', va='center')

        for bx in [ax1, ax2]:
            bx.set(ylabel=None)

        ax2.axes.yaxis.set_ticklabels([])

        ax1.axvspan(np.median(data_wp_ltr[-1]), np.median(data_wp_ltr[-2]), color='#909090', zorder=-1, alpha=0.3)
        ax2.axvspan(np.median(data_ep_ltr[-1]), np.median(data_ep_ltr[-2]), color='#909090', zorder=-1, alpha=0.3)

        ax1.set_title('(c) West Pacific', fontsize=12)
        ax2.set_title('(d) Central-East Pacific', fontsize=12)

        # 1979-2014 trends
        ymin = np.linspace(0.01, 0.89, 8)
        [ax1.axvline(data_wp_tt[::-1][i], ymin=ymin[i], ymax=ymin[i] + 0.1, color=PALETTE[::-1][i], linewidth=2)
        for i in range(len(data_wp_tt))]
        [ax2.axvline(data_ep_tt[::-1][i], ymin=ymin[i], ymax=ymin[i] + 0.1, color=PALETTE[::-1][i], linewidth=2)
        for i in range(len(data_ep_tt))]

        ax1.text(0.85, -0.2, '$^\circ$C $decade^{-1}$', fontsize=11, transform=ax1.transAxes)

        plt.suptitle('Equatorial SST trends for 30-65 year periods', y=0.95, fontsize=11)
        plt.savefig(filename, dpi=750, facecolor='white', bbox_inches="tight")
        print(f"Long-term trends plot saved to {filename}")

    except Exception as e:
        print(f"Error in plot_longterm_trends: {e}")

# ============================================================
### Execute script

if __name__ == "__main__":
    data_tauu = [ll_tauu, mm_tauu, hm_tauu, hh_tauu, mlr_tauu, mhr_tauu, mxr_tauu, mer_tauu]
    data_theta = [ll_theta, mm_theta, hm_theta, hh_theta, mlr_theta, mhr_theta, mxr_theta, mer_theta]

    ws_bias, d20m_bias, d20s_bias = calculate_mean_windstress_thermocline_biases(data_tauu, data_theta)
    plot_mean_windstress_thermocline_bias(ws_bias, d20m_bias, d20s_bias, filename='Fig5_1.png')

    # Data for trends
    data = [ll_tos_dc, mm_tos_dc, hm_tos_xr, mlr_tos_dc, mhr_tos_dc, mer_tos_dc, ersst_tos, cobe_tos]

    # Calculate Regional Means
    data_ep = [calculate_regional_mean(v, LAT_MIN, LAT_MAX, EP_LON_MIN, EP_LON_MAX) for v in data]
    data_wp = [calculate_regional_mean(v, LAT_MIN, LAT_MAX, WP_LON_MIN, WP_LON_MAX) for v in data]

    # Calculate Long Term Trends
    data_ep_lt = [long_term_trends(v, v.time, START_YEAR, END_YEAR) for v in data_ep]
    data_wp_lt = [long_term_trends(v, v.time, START_YEAR, END_YEAR) for v in data_wp]

    # Select random trends
    data_ep_ltr = [np.random.choice(v, size=595, replace=False) for v in data_ep_lt]
    data_wp_ltr = [np.random.choice(v, size=595, replace=False) for v in data_wp_lt]

    # Create DataFrames for Plotting
    df_data_ep = create_dataframe(data_ep_lt, MODEL_NAMES)
    df_data_wp = create_dataframe(data_wp_lt, MODEL_NAMES)

    # Calculate Trends for 1979-2014
    data_ep_t = [v.sel(time=slice(str(START_YEAR), str(END_YEAR))) for v in data_ep]
    data_wp_t = [v.sel(time=slice(str(START_YEAR), str(END_YEAR))) for v in data_wp]
    data_ep_tt = [mk_test(v)[0] * 10 for v in data_ep_t] 
    data_wp_tt = [mk_test(v)[0] * 10 for v in data_wp_t] 

    plot_longterm_trends(df_data_wp, df_data_ep, data_wp_tt, data_ep_tt, data_wp_ltr, data_ep_ltr, filename='Fig5_2.png')