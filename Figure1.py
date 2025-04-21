#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script analyzes and visualizes sea surface temperature (SST) biases
in climate models compared to observations, and their relationship with the 
tropical Pacific SST trend discrepancy, focusing on key regions.
"""

__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
## Load modules
import warnings
import itertools

# Local modules
from import_highresmip_data import *
from help_func import *
from plot_map import *

# Suppress warnings
warnings.filterwarnings('ignore')

# Set matplotlib font size
plt.rc('font', size=12)

# Set global grid settings for proplot
pplt.rc.update(grid=False)

#============================================================
#Constants 
START_YEAR = 1979
END_YEAR = 2014

COLORS = ["#0072B2", "#332288", "#D55E00", "#AA4499", "#009E73", "#56B4E9", "#2f2f2f"]
COLORS_RPT = [color * count for color, count in zip(COLORS, [2, 2, 3, 3, 3, 4, 4])]
COLORS_MODEL = ["#0072B2", "#332288", "#D55E00", "#AA4499", "#009E73", "#56B4E9", "#2f2f2f"]
MODEL_LABELS = ['MPI-ESM1-2', 'HadGEM3-GC31', 'ECMWF-IFS', 'EC-Earth3P', 'CMCC-CM2', 'CNRM-CM6-1', 'CESM1-CAM5']

CMAP = pplt.Colormap("ColdHot")
NAN_COLOR = 'lightgrey'
ROBINSON_CENTRAL_LON = 180
LINEWIDTH = 0.3
CONTOUR_COLOR = '#5D3A9B'
CONTOUR_FONTSIZE = 6
DPI = 750

# Region Definitions
COLD_TONGUE_LATS = [-5, -5, 5, 5, -5]
COLD_TONGUE_LONS = [-95, 180, 180, -95, -95]
SEP_WARM_BIAS_LATS = [-20, -20, 0, 0, -20]
SEP_WARM_BIAS_LONS = [-70, -95, -95, -70, -70]
EASTERN_ATLANTIC_LATS = [-20, -20, 5, 5, -20]
EASTERN_ATLANTIC_LONS1 = [-20, 359, 359, -20, -20]
EASTERN_ATLANTIC_LONS2 = [0, 10, 10, 0, 0]
SOUTHERN_OCEAN_LATS = [-75, -75, -45, -45, -75]
SOUTHERN_OCEAN_LONS = [359, 0, 0, 359, 359]

#============================================================
def calculate_tos_bias(models_tos, obs_tos):
    """Calculates the SST bias between a list of model outputs and observations."""
    stacked_models = np.stack([np.nanmean(v, axis=(0, 1)).squeeze() for v in models_tos])
    mean_model = np.nanmean(stacked_models, axis=0)
    mean_obs = np.nanmean(obs_tos, axis=(0, 1))
    return mean_model - mean_obs

def plot_contour(ax, lon, lat, obs_tos):
    clevels = [20, 27, 29]
    # Contour plots
    cs = ax.contour(lon, lat, obs_tos.mean(dim=('time')), levels=clevels,
                    colors=CONTOUR_COLOR, linewidths=LINEWIDTH, transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=1, fontsize=CONTOUR_FONTSIZE)

def plot_regions(ax):
    """Plots predefined regions on the given axes."""
    regions = [
        {'lons': COLD_TONGUE_LONS, 'lats': COLD_TONGUE_LATS, 'style': {'color': 'black', 'linewidth': 0.6}},
        {'lons': SOUTHERN_OCEAN_LONS, 'lats': SOUTHERN_OCEAN_LATS, 'style': {'color': 'black', 'linewidth': LINEWIDTH, 'ls': '--'}},
        {'lons': SEP_WARM_BIAS_LONS, 'lats': SEP_WARM_BIAS_LATS, 'style': {'color': 'black', 'linewidth': LINEWIDTH, 'ls': '--'}},
        {'lons': EASTERN_ATLANTIC_LONS1, 'lats': EASTERN_ATLANTIC_LATS, 'style': {'color': 'black', 'linewidth': LINEWIDTH, 'ls': '--'}},
        {'lons': EASTERN_ATLANTIC_LONS2, 'lats': EASTERN_ATLANTIC_LATS, 'style': {'color': 'black', 'linewidth': LINEWIDTH, 'ls': '--'}}
    ]
    # Plot the regions using a loop
    for region in regions:
        print([x%360 for x in region['lons']])
        ax.plot([x%360 for x in region['lons']], region['lats'],
                transform=ccrs.PlateCarree(), **region['style'])

def calculate_linear_regression(data1, data2):
    """
    Calculates the slope, R-squared, and p-value of a linear regression 
    between remote SST biases (data2) and 
    cold tongue bias/equatorial east Pacific trend discrepancy/southeast Pacific trend discrepancy (data1).

    Args:
        data1 (np.ndarray): Cold tongue bias/trend discrepancy data.
        data2 (np.ndarray): Remote SST bias data.

    Returns:
        tuple: slope, R-squared, p-value arrays.
    """

    y = data1[:-1] - data1[-1]
    X = (np.stack(data2[:-1]) - np.array(data2[-1])).mean(axis=1)

    # Initialize output arrays with the correct shape
    slope = np.empty(data2[-1][0].shape)
    r_squared = np.empty(data2[-1][0].shape)
    p = np.empty(data2[-1][0].shape)

    # Loop over indices and perform regression
    for i in range(slope.shape[0]):
        for j in range(slope.shape[1]):
            xi = np.nan_to_num(X[:, i, j].reshape(-1, 1))

            # Check for variance before linear regression
            if np.var(xi) == 0:  # No variance in the independent variable
                slope[i, j] = np.nan
                r_squared[i, j] = np.nan
                p[i, j] = np.nan 
                continue

            # Linear regression using scipy.stats.linregress
            slope[i, j], _, r_value, p[i, j], _ = scipy.stats.linregress(xi.flatten(), y)
            r_squared[i, j] = r_value**2  # Calculate R-squared from r_value

    # Clean up: replace zeros with NaN
    slope[slope == 0.0] = np.nan
    r_squared[r_squared == 0.0] = np.nan
    p[p == 0.0] = np.nan

    # Set p-values > 0.05 to NaN
    p[p > 0.05] = np.nan

    # Create a boolean mask for NaN values in p_ct
    p_ma = np.isnan(p)

    return slope, r_squared, p, p_ma

def calculate_regional_bias(data_em, lat, lon, region_def):
    """
    Calculates the weighted mean SST for a specified region.

    Args:
        data_em (list of xr.DataArray): List of SST data for different models.
        lat (xr.DataArray): Latitude data.
        lon (xr.DataArray): Longitude data.
        region_def (tuple): Latitude and longitude boundaries (lat1, lat2, lon1, lon2).

    Returns:
        np.ndarray: Array of weighted mean SST values for the specified region.
    """
    lat1, lat2, lon1, lon2 = region_def

    # Select data for the region
    data_reg = [v.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)) for v in data_em]
    data_reg = np.stack(data_reg)

    # Select corresponding latitudes and longitudes
    lats = lat.sel(lat=slice(lat1, lat2))
    lons = lon.sel(lon=slice(lon1, lon2))

    # Calculate weighted mean SST
    data_regw = np.array([wgt_mean(v, lons, lats) for v in data_reg])

    return data_regw

#============================================================
#Main plotting functions 

def plot_bias_maps(mods_tos, obs_tos, lon, lat, filename="SST_Bias_Maps.png"):
    """
    Generates and saves maps of SST bias, comparing low and high-resolution models.

    Args:
        mods_tos (list): List of model SST data (xarray DataArrays).
        obs_tos (xarray.DataArray): Observed SST data.
        start_year (int): Start year for the analysis.
        end_year (int): End year for the analysis.
        lon (np.ndarray): Longitude coordinates.
        lat (np.ndarray): Latitude coordinates.
        filename (str): The name of the output file.
    """
    try:
        # Calculate average bias
        tos_bias_lowres = tos_bias_lowres = calculate_tos_bias([v for v in mods_tos[13:]], obs_tos)
        tos_bias_highres = calculate_tos_bias([v for v in mods_tos[:13]], obs_tos)

        # Plot setup
        fig, axs, ax_cb = setup_map_plot()
        var = [tos_bias_lowres, tos_bias_highres]
        titles = ["(a) Lower resolution", "(b) Higher resolution"]
        
        levels = np.arange(-3, 3.5, 0.5)
        clevels = [20, 27, 29]

        for i, ax in enumerate(axs):
            plot_map_gridspec(
                ax,
                var[i],
                lon,
                lat,
                levels=levels,
                cmap=CMAP,
                ticks=False,
                land=True,
                title=titles[i],
                loc_title="center",
                pad=5,
            )

            #Plot countours
            cs = ax.contour(
                lon,
                lat,
                obs_mean,
                levels=clevels,
                colors=CONTOUR_COLOR,
                linewidths=LINEWIDTH,
                transform=ccrs.PlateCarree(),
            )
            ax.clabel(cs, inline=1, fontsize=CONTOUR_FONTSIZE)

            #Plot regions
            plot_regions(ax)

        # Colorbar
        create_colorbar(ax_cb, cmap, levels, label='$^\circ$C')

        plt.suptitle("Mean state SST bias", fontsize=11, y=0.96)
        plt.savefig(filename, dpi=DPI, bbox_inches="tight")

        print(f"SST Bias Maps plot saved to {filename}")

    except Exception as e:
        print(f"Error in plot_bias_maps: {e}")

def plot_cold_tongue_bias(data, lat, lon, mlabels, colors, markers, filename="Cold_Tongue_Bias.png"):
    """Plots cold tongue bias for different models."""
    try:
        fig, ax = plt.subplots(figsize=(7.48, 2.5))
        pplt.rc.update({"xtick.minor.visible": False})

        lat1, lat2, lon1, lon2 = cold_tongue_region
        lats = lat.sel(lat=slice(lat1, lat2))
        lons = lon.sel(lon=slice(lon1, lon2))

        data_reg = [v.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)) for v in data]
        x_w = np.array([wgt_mean(v, lons, lats) for v in np.concatenate(data_reg)])

        yloc = np.arange(len(data_reg))[::-1]
        y = np.repeat(yloc, [len(d) for d in data_reg])
        c = np.repeat(colors, [len(d) for d in data_reg])
        m = np.repeat(markers[:len(data) - 1], [len(d) for d in data_reg[:-1]]).squeeze()

        tp = x_w[:-1] - x_w[-1]

        for i in range(len(x_w[:-1])):  # for each of the features
            mi = m[i]  # marker for ith feature
            yi = y[i]  # y array for ith feature
            ci = c[i]  # color for ith feature
            ax1.scatter(yi, tp[i], marker=mi, color=ci, s=45, facecolors="none")

        ax.set_xticklabels(mlabels[::-1], fontsize=9)
        ax.set_title("(c) Cold tongue bias", fontsize=12)
        ax.set_ylabel('$^\circ$C', size=10)
        ax.set_xticks(np.arange(len(data_reg)))

        lines = [4.5, 8.5, 11.5, 14.5, 17.5, 19.5, 21.5]
        ax.set_xlim(0.5, len(data) - 0.5)
        ax.invert_xaxis()
        ax.axvline(lines, linestyle="--", color="k", lw=0.5, alpha=0.8)
        ax.axhline(0.0, color="k", lw=0.5, zorder=-1)
        ax.axvspan([0.5, 1.5, 4.5, 5.5, 6.5, 8.5, 9.5, 11.5, 12.5, 14.5, 15.5, 17.5, 19.5],
                   [1.5, 2.5, 5.5, 6.5, 7.5, 9.5, 10.5, 12.5, 13.5, 15.5, 16.5, 18.5, 20.5],
                   facecolor="k", alpha=0.1)

        y = -2.1
        for line, name, color in zip(lines, MODEL_LABELS, COLORS_MODEL[::-1]):
            ax.text(line - (2 if name in ["MPI-ESM1-2", "HadGEM3-GC31"] else 1.5), y, name,
                    fontsize=9, color=color, ha="center", va="center")

        plt.savefig(filename, dpi=DPI, bbox_inches="tight")
        print(f"Cold Tongue Bias plot saved to {filename}")

    except Exception as e:
        print(f"Error in plot_cold_tongue_bias: {e}")

def plot_eddy_bias_maps(pst_tos, rch_tos, obs_tos, lon, lat, filename="Eddy_Bias_Maps.png"):
    """
    Generates and saves maps of SST bias for eddy-present and eddy-rich models.

    Args:
        mods_pst (list): List of eddy-present model SST data (xarray DataArrays).
        mods_rch (list): List of eddy-rich model SST data (xarray DataArrays).
        obs_tos (xarray.DataArray): Observed SST data.
        start_year (int): Start year for the analysis.
        end_year (int): End year for the analysis.
        lon (np.ndarray): Longitude coordinates.
        lat (np.ndarray): Latitude coordinates.
        filename (str): The name of the output file.
    """
    try:
        tos_bias_pst = calculate_tos_bias(pst_tos, obs_tos)
        tos_bias_rch = calculate_tos_bias(rch_tos, obs_tos)

        #Plot setup
        fig = plt.figure(figsize=(7.48, 2.5))
        spec = gridspec.GridSpec(
            ncols=2,
            nrows=2,
            width_ratios=[1, 1],
            height_ratios=[1, 0.04],
            hspace=0.0,
            wspace=0.05,
        )
        projection = ccrs.Robinson(ROBINSON_CENTRAL_LON)
        ax1 = fig.add_subplot(spec[0, 0], projection=projection)
        ax2 = fig.add_subplot(spec[0, 1], projection=projection)
        axs = [ax1, ax2]
        ax_cb = fig.add_subplot(spec[1, :])

        # Set global settings to remove minor ticks
        pplt.rc.update({'xtick.minor.visible': True})

        titles = ["(a) Eddy-present", "(b) Eddy-rich"]
        var = [tos_bias_pst, tos_bias_rch]
        levels = np.arange(-3, 3.5, 0.5)
        clevels = [20, 27, 29]
        cmap = pplt.Colormap("ColdHot")

        for i in range(len(axs)):
            plot_maps_gridspec(
                axs[i],
                var[i],
                lon,
                lat,
                levels=levels,
                mp=0,
                cmap=cmap
                ticks=False,
                land=True,
                title=titles[i],
                loc_title="center",
                pad=5,
            )

            cs = axs[i].contour(
                lon,
                lat,
                obs_mean,
                levels=clevels,
                colors="#5D3A9B",
                linewidths=0.3,
                transform=ccrs.PlateCarree(),
            )
            axs[i].clabel(cs, inline=1, fontsize=6)

            plot_regions(axs[i])

        create_colorbar(ax_cb, cmap, levels, label='$^\circ$C')

        plt.suptitle("Mean state SST bias", fontsize=11, y=0.96)
        plt.savefig(filename, dpi=DPI, bbox_inches="tight")
        print(f"Eddy Bias Maps plot saved to {filename}")

    except Exception as e:
        print(f"Error in plot_eddy_bias_maps: {e}")

def plot_regional_biases(data, lat, lon, mlabels, colors, markers, regions_def, titles, filename="Regional_Biases.png"):
    """Plots average mean state SST biases for the specified regions."""
    try:
        fig, axs = plt.subplots(1, 3, figsize=(7.48, 5.48), gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.05})

        pplt.rc.update({"ytick.minor.visible": False, "xtick.minor.visible": True})

        for idx, ax in enumerate(axs):
            lat1, lat2, lon1, lon2 = regions_def[idx]
            lats = lat.sel(lat=slice(lat1, lat2))
            lons = lon.sel(lon=slice(lon1, lon2))

            data_reg = [v.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)) for v in data]
            x_reg = np.concatenate(data_reg)
            x_w = np.array([wgt_mean(v, lons, lats) for v in x_reg])

            yloc = np.arange(len(data_reg))[::-1]
            y = np.repeat(yloc, [len(d) for d in data_reg])
            c = np.repeat(colors, [len(d) for d in data_reg])
            m = np.repeat(markers[:len(data)-1], [len(d) for d in data_reg[:-1]]).squeeze()

            tp = x_w[:-1] - x_w[-1]

            for i in range(len(x_w[:-1])):  # for each of the models
                mi = m[i]  # marker for ith feature
                yi = y[i]  # y array for ith feature
                ci = c[i]  # color for ith feature
                ax.scatter(tp[i], yi, marker=mi, color=ci, s=45, facecolors="none")

            ax.set_yticks(np.arange(len(data_reg)))
            ax.set_yticklabels("")
            ax.set_title(titles[idx])
            ax.set_xlabel('$^\circ$C', size=9)
            ax.axvline(0, color='k', lw=0.5, zorder=-1)

            lines = [4.5,8.5,11.5,14.5,17.5,19.5,21.5]
            ax.set_ylim(0.5, 21.5)
            ax.invert_yaxis()
            ax.axhline(lines, linestyle='--', color='k', lw=0.5, alpha=0.8)
            ax.axvspan([0.5, 1.5, 4.5, 5.5, 6.5, 8.5, 9.5, 11.5, 12.5, 14.5, 15.5, 17.5, 19.5], 
                       [1.5, 2.5, 5.5, 6.5, 7.5, 9.5, 10.5, 12.5, 13.5, 15.5, 16.5, 18.5, 20.5], 
                       facecolor='k', alpha=0.1)

        #Model labels
        for i, (name, color, y) in enumerate(zip(MODEL_LABELS, COLORS_MODEL, lines)):
            axs[0].text(-1.8, y - (2 if i < 2 else 1.5), name, rotation='vertical', fontsize=8, 
                        color=color, ha='center', va='center')

        axs[0].set_yticklabels(mlabels[::-1], fontsize=9)
        
        plt.suptitle("Mean state SST bias", fontsize=11, y=0.96)
        plt.savefig(filename, dpi=DPI, bbox_inches="tight")
        print(f"Regional Biases plot saved to {filename}")

    except Exception as e:
        print(f"Error in plot_regional_biases: {e}")

def plot_cold_tongue_remote_biases(slope, r_squared, p_ma, lon, lat, filename="Remote_Local_Bias_Maps.png"):
    """Plots the relationship between remote mean state SST biases and the cold tongue bias."""

    try:
        # Plot setup
        fig = plt.figure(figsize=(7.48, 2.24))
        spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=[1, 1],
                                 height_ratios=[1, 0.04], hspace=0.1, wspace=0.1)
        projection = ccrs.Robinson(ROBINSON_CENTRAL_LON)
        axs = [fig.add_subplot(spec[0, i], projection=projection) for i in range(2)]
        ax_cb1 = fig.add_subplot(spec[1, 0])
        ax_cb2 = fig.add_subplot(spec[1, 1])

        titles = ['(a)', '(b)']
        var = [slope, r_squared]
        pval = [p_ma, p_ma]

        levels = [np.arange(-1, 1.1, 0.1), np.arange(0, 1, 0.1)]
        mps = [0.0, 0.5]
        cmap = [pplt.Colormap('ColdHot'), pplt.Colormap('glacial')]

        # Helper function for plotting regions
        def plot_regions(ax):
            """Plots predefined regions on the given axes."""
            regions = [
                {"lons": [-95, 180, 180, -95, -95], "lats": [-5, -5, 5, 5, -5], "style": {"color": "black", "linewidth": 0.6}},
                {"lons": [-60, -15, -15, -60, -60], "lats": [0, 0, 15, 15, 0], "style": {"color": "black", "linewidth": 0.3, "ls": "--"}},
                {"lons": [30, 110, 110, 30, 30], "lats": [-25, -25, 25, 25, -25], "style": {"color": "black", "linewidth": 0.3, "ls": "--"}},
                {"lons": [360, 0, 0, 360, 360], "lats": [-75, -75, -45, -45, -75], "style": {"color": "black", "linewidth": 0.3, "ls": "--"}}
            ]

            for region in regions:
                ax.plot([x % 360 for x in region["lons"]], region["lats"],
                        transform=ccrs.PlateCarree(), **region["style"])

        # Plotting loop
        for i, ax in enumerate(axs):
            plot_map_gridspec(ax, var[i], lon, lat, levels=levels[i], mp=mps[i],
                              cmap=cmap[i], ticks=False, land=True, title=titles[i],
                              loc_title='center', fontsize=12, pad=5)
            ax.contour(lon, lat, pval[i], [1], colors='k', linewidths=0.5,
                       transform=ccrs.PlateCarree())
            plot_regions(ax)

        # Colorbar
        create_colorbar(ax_cb1, cmap[0], levels[0], label='$decade^{-1}$')
        create_colorbar(ax_cb2, cmap[1], levels[1], label='')

        # Annotations
        plt.suptitle('Relationship between remote mean state SST biases and cold tongue bias (1979 - 2014)',
                     fontsize=11, y=1.12)
        ax1.text(0.5, 1.22, 'Slope', transform=ax1.transAxes, fontsize=12, ha='center')
        ax2.text(0.5, 1.22, 'Coefficient of determination', transform=ax2.transAxes, fontsize=12, ha='center')

        plt.savefig(filename, dpi=DPI, bbox_inches="tight")

        print(f"Remote-local Bias Maps plot saved to {filename}")

    except Exception as e:
        print(f"Error in plot_cold_tongue_remote_biases: {e}")

def plot_cold_tongue_remote_biases_scatter(data_tio, data_nao, data_so, data_ct, filename="Remote_Local_Bias_Scatter.png"):
    """Plots the relationship between remote mean state SST biases for key focused regions and the cold tongue bias."""
    try:
        # Plot setup
        fig = plt.figure(figsize=(7.48,2.74))
        spec = gridspec.GridSpec(ncols=3, nrows=1, width_ratios=[1,1,1],
                                height_ratios=[1]],hspace=0.,wspace=0.1)
        axs = [fig.add_subplot(spec[0, i]) for i in range(3)]
    
        # Data preparation
        x = [data_tio[:-1] - data_tio[-1],
            data_nao[:-1] - data_nao[-1],
            data_so[:-1] - data_so[-1]]
        y = data_ct[:-1] - data_ct[-1]
        titles = ['(c)', '(d)', '(e)']
        xlabels = ['Indian Ocean bias [$^\circ$C]',
                'Western tropical\nAtlantic Ocean bias [$^\circ$C]',
                'Southern Ocean bias [$^\circ$C]']
        xlims = [(-1.2, 0.5), (-1.2, 0.5), (-0.5, 2)]
    
        # Model Colors and Labels
        s = np.arange(1,len(COLORS)+1)
        sizes = [(s[i]+3)**2 for i in range(len(s))][::-1]

        # Plotting Loop
        for n, ax in enumerate(axs):
            # Scatter plot
            for i in range(len(COLORS)):
                ax.scatter(x[n][i], y[i], s=sizes[i], marker='o', lw=1.5, facecolors='none', edgecolors=COLORS[i])

            ax.set_title(titles[n], pad=-0.2)

            # Calculate and display correlation coefficient
            r = np.corrcoef(x[n], y)[0, 1]  # Corrected correlation calculation
            ax.text(0.1, 0.92, f'r = {np.round(r, 1)}', transform=ax.transAxes, fontsize=12)

            # Linear regression line
            b, a = np.polyfit(x[n], y, deg=1)
            xseq = np.linspace(np.nanmin(x[n]), np.nanmax(x[n]), num=100)
            ax.plot(xseq, a + b * xseq, color="#909090", lw=0.4, ls='--')

            # Zero lines and Spines
            ax.axhline(0.0, color='#909090', zorder=-1)
            ax.axvline(0.0, color='#909090', zorder=-1)
            ax.spines[['right', 'top']].set_visible(False)

            # Limits and Labels
            ax.set_ylim(-1.5, 0.5)
            ax.set_xlim(xlims[n])
            if n > 0:
                ax.set_yticklabels([])
            if n == 0:
                ax.set_ylabel('Cold tongue bias [$^\circ$C]')
            ax.set_xlabel(xlabels[n])

        # Model Labels (Rightmost Plot)
        x_pos = 1.5
        y_start = -1.4
        y_increment = 0.2
        for i, label in enumerate(MODEL_LABELS):
            axs[2].text(x_pos, y_start + (y_increment * i), label, fontsize=10, color=COLORS_MODEL[i])

        plt.tight_layout()
        plt.savefig(filename, dpi=DPI, bbox_inches="tight")

        print(f"Remote-local Bias Scatter plot saved to {filename}")

    except Exception as e:
        print(f"Error in plot_cold_tongue_remote_biases_scatter: {e}")

def plot_pacific_trend_discrepancy_remote_biases(slope_ep, r_squared_ep, p_ma_ep,
                                                 slope_sep, r_squared_sep, p_ma_sep, lon, lat,
                                                 filename="Remote_Bias_Local_Discrepancy_Maps.png"):
    """Plots the relationship between remote SST biases and the tropical Pacific SST trend discrepancy."""
    try:
        # Plot setup
        fig = plt.figure(figsize=(7.48, 4.48))
        spec = gridspec.GridSpec(ncols=2, nrows=3, width_ratios=[1, 1],
                                 height_ratios=[1, 1, 0.04], hspace=0.1, wspace=0.1)
        projection = ccrs.Robinson(ROBINSON_CENTRAL_LON)
        axs = [fig.add_subplot(spec[i], projection=projection) for i in [0, 1, 2, 3]]
        ax_cb1 = fig.add_subplot(spec[4])
        ax_cb2 = fig.add_subplot(spec[5])

        titles = ['(a)', '(b)', '(c)', '(d)']
        var = [slope_ep, r_squared_ep, slope_sep, r_squared_sep]
        pval = [p_ma_ep, p_ma_ep, p_ma_sep, p_ma_sep]

        levels = [np.arange(-0.06, 0.07, 0.01), np.arange(0, 1, 0.1),
                  np.arange(-0.06, 0.07, 0.01), np.arange(0, 1, 0.1)]
        mps = [0.0, 0.5, 0.0, 0.5]

        cmap = [pplt.Colormap('ColdHot'), pplt.Colormap('glacial')] * 2  # Replicated for 4 plots

        # Helper function for plotting regions
        def plot_regions(ax, region_type):
            """Plots predefined regions on the given axes."""
            if region_type == "East Pacific":
                lons = [-180, -95, -95, -180, -180]
                lats = [-5, -5, 5, 5, -5]
            elif region_type == "Southeast Pacific":
                # Define regions as list of dictionaries
                regions = [
                    {"lons": [-70, -70, -160, -160, -70], "lats": [-45, -45, -5, -5, -45], "style": {"linewidth": 1}},
                    {"lons": [40, -160, -160, 40, 40], "lats": [-25, -25, 25, 25, -25], "style": {"linewidth": 0.3, "ls": "--"}},
                    {"lons": [360, 0, 0, 360, 360], "lats": [-75, -75, -45, -45, -75], "style": {"linewidth": 0.3, "ls": "--"}},
                ]
                for region in regions:
                    ax.plot([x % 360 for x in region["lons"]], region["lats"], transform=ccrs.PlateCarree(), color='black', **region["style"])
                return 
            else:
                raise ValueError("Invalid region_type. Must be 'East Pacific' or 'Southeast Pacific'.")

            ax.plot([x % 360 for x in lons], lats, color='black', linewidth=1, marker='', transform=ccrs.PlateCarree())

        # Plotting loop
        for i, ax in enumerate(axs):
            plot_maps_gridspec(ax, var[i], lon, lat, levels=levels[i], mp=mps[i],
                              cmap=cmap[i], ticks=False, land=True, title=titles[i],
                              loc_title='center', fontsize=12, pad=5)
            ax.contour(lon, lat, pval[i], [1], colors='k', linewidths=0.5,
                       transform=ccrs.PlateCarree())

        [plot_regions(ax, "East Pacific") for ax in [axs[0],axs[1]]] 
        [plot_regions(ax, "Southeast Pacific") for ax in [axs[2],axs[3]]] 

        # --- Colorbar ---
        create_colorbar(ax_cb1, cmap[0], levels[0], label='$decade^{-1}$')
        create_colorbar(ax_cb2, cmap[1], levels[1], label='')

        # --- Annotations ---
        plt.suptitle('Relationship between remote mean state SST bias and\ntropical Pacific SST trend discrepancy (1979 - 2014)',
                     fontsize=11, y=1.04)

        axs[0].text(-0.1, 0.5, 'Central-to-eastern\nequatorial Pacific', transform=axs[0].transAxes,
                 fontsize=12, va='center', ha='center', rotation='vertical')
        axs[2].text(-0.1, 0.5, 'Southeast Pacific', transform=axs[2].transAxes,
                 fontsize=12, va='center', rotation='vertical')

        axs[0].text(0.5, 1.22, 'Slope', transform=axs[0].transAxes, fontsize=12, ha='center')
        axs[1].text(0.5, 1.22, 'Coefficient of determination', transform=axs[1].transAxes, fontsize=12, ha='center')
        axs[2].text(0.5, 1.22, 'Slope', transform=axs[2].transAxes, fontsize=12, ha='center')
        axs[3].text(0.5, 1.22, 'Coefficient of determination', transform=axs[3].transAxes, fontsize=12, ha='center')

        plt.savefig(filename, dpi=DPI, bbox_inches="tight")

        print(f"Remote Bias Local Trend Discrepancy Maps plot saved to {filename}")

    except Exception as e:
        print(f"Error in plot_pacific_trend_discrepancy_remote_biases: {e}")

def plot_pacific_trend_discrepancy_remote_biases_scatter(data_ipwp, data_so, tdata_sep, filename="Remote_Bias_Local_Discrepancy_Maps.png"):
    """Plots the relationship between remote SST biases for key focused regions and the tropical Pacific SST trend discrepancy."""
    try:
        # Plot setup
        fig = plt.figure(figsize=(7.48,3.74))
        spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1,1],
                                height_ratios=[1]],hspace=0.,wspace=0.1)
        axs = [fig.add_subplot(spec[0, i]) for i in range(2)]

        # Data preparation
        x = [data_ipwp[:-1] - data_ipwp[-1], 
            data_so[:-1] - data_so[-1]]
        y = tdata_sep[:-1] - tdata_sep[-1]
        titles = ['(e)', '(f)']
        xlabels = ['Indo Pacific Warm Pool bias [$^\circ$C]', 'Southern Ocean bias [$^\circ$C]']

        for n, (ax, x, title, xlabel) in enumerate(zip(axs, x_data, titles, xlabels)):
            for i, color in enumerate(COLORS):
                ax.scatter(x[i], y[i], s=sizes[i], marker='o', lw=1.5, facecolors='none', edgecolors=color)

            ax.set_title(title, pad=-0.2)

            r = np.round(np.corrcoef(x.data, y.data)[1][0], 1)
            ax.text(0.75, 0.92, f'r = {r}', transform=ax.transAxes, fontsize=12)

            # Fit and plot linear regression line
            b, a = np.polyfit(x, y, deg=1)
            xseq = np.linspace(np.nanmin(x), np.nanmax(x), num=100)
            ax.plot(xseq, a + b * xseq, color="#909090", lw=0.4, ls='--')
            
            # Zero lines
            ax.axhline(0., color='#909090', zorder=-1)
            ax.axvline(0., color='#909090', zorder=-1)
            
            # Hide the right and top spines
            ax.spines[['right', 'top']].set_visible(False)
            
            ax.set_xlabel(xlabel)

        # Labels
        axs[0].set_ylabel('Southeast Pacific\nSST trend discrepancy [$^\circ$C $decade^{-1}$]')
        axs[1].set_yticklabels([])

        # Model Labels (Rightmost Plot)
        x_pos = 1.7
        y_start = 0
        y_increment = 0.025
        for i, label in enumerate(MODEL_LABELS):
            axs[1].text(x_pos, y_start + (y_increment * i), label, fontsize=10, color=COLORS_MODEL[i])

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Remote bias-local discrepancy Scatter plot saved to {filename}")

    except Exception as e:
        print(f"Error in plot_pacific_trend_discrepancy_remote_biases_scatter: {e}")

# ============================================================
### Execute script

if __name__ == "__main__":

    # All models data setup according to increasing resolution
    mdata_res = [
        ecvhr_tos,
        hh_tos,
        nhr_tos,
        mer_tos,
        cmvhr_tos,
        ehr_tos,
        emr_tos,
        hm_tos,
        echr_tos,
        cmhr_tos,
        cnhr_tos,
        mm_tos,
        mxr_tos,
        mhr_tos,
        elr_tos,
        cmsr_tos,
        eclr_tos,
        nle_tos,
        ll_tos,
        cnlr_tos,
        mlr_tos,
    ]  # All models

    #Select time
    years = mlr_tos_dc.time.dt.year
    obs_tos = ersst_tos.sel(time=slice(str(START_YEAR), str(END_YEAR)))
    mods_tos = [seltime(v, years, START_YEAR, END_YEAR) for v in mdata_res]

    obs_mean = np.nanmean(obs_tos, axis=(0, 1))

    #Plot SST bias maps
    plot_bias_maps(mods_tos, obs_tos, lon, lat, filename='Fig1_1.png')

    # ============================================================
    #All data setup according to model group
    data = [
        nle_tos,
        nhr_tos,
        cnlr_tos,
        cnhr_tos,
        cmsr_tos,
        cmhr_tos,
        cmvhr_tos,
        eclr_tos,
        echr_tos,
        ecvhr_tos,
        elr_tos,
        emr_tos,
        ehr_tos,
        ll_tos,
        mm_tos,
        hm_tos,
        hh_tos,
        mlr_tos,
        mhr_tos,
        mxr_tos,
        mer_tos,
        obs_tos,
    ]

    mlabels = [
        "LR",
        "HR",
        "LR",
        "HR",
        "SR5",
        "HR4",
        "VHR4",
        "LR",
        "HR",
        "VHR",
        "LR",
        "MR",
        "HR",
        "LL",
        "MM",
        "HM",
        "HH",
        "LR",
        "HR",
        "XR",
        "ER",
        "",
    ]

    nmod = len(data) - 1
    colors_rpt = list(itertools.chain.from_iterable(COLORS_RPT))  # Flatten
    markers = ['o'] * (nmod + 1)
    
    #Select time
    time = mlr_tos.time
    reftime = time.sel(time=slice(str(START_YEAR), str(END_YEAR)))
    ser = np.arange(len(reftime))
    mdata = [seltime(v, years, START_YEAR, END_YEAR) for v in data[:-1]]
    data = mdata + [obs_tos]
    data_em = [v.mean(axis=0) for v in data]

    # Define region of interest
    cold_tongue_region = (-5, 5, 180, -95 % 360)

    # Plot cold tongue bias
    plot_cold_tongue_bias(data, lon, lat, mlabels, colors_rpt, markers, filename='Fig1_2.png')`

    # ============================================================
    #Model data setup according to eddy resolution
    pst_tos = [ecvhr_tos, hh_tos, nhr_tos, mer_tos]  # Eddy-present models
    rch_tos = [cmvhr_tos, ehr_tos, emr_tos, hm_tos, echr_tos, cmhr_tos, cnhr_tos, mm_tos]  # Eddy-rich models

    #Select time 
    pst_tos = [seltime(v, years, START_YEAR, END_YEAR) for v in pst_tos]
    rch_tos = [seltime(v, years, START_YEAR, END_YEAR) for v in rch_tos]

    plot_eddy_bias_maps(pst_tos, rch_tos, obs_tos, lon, lat, filename="SFig1.png")

    # ============================================================
    # Define regions for regional bias plots
    regions_def = [
        (-20, -0, -95%360, -70%360),  # Southeast Pacific
        (-20, 5, -25%360, 360),  # Eastern Atlantic
        (-75, -45, -140%360, -80%360)]  # Southern Ocean

    titles = ["(a) Southeastern Pacific", "(b) Eastern Atlantic", "(c) Southern Ocean"]

    plot_regional_biases(data, lat, lon, mlabels, COLORS, markers, regions_def, titles, filename="SFig3.png")

    # ============================================================
    # Define regions of interest
    southern_ocean_region = (-75, -45, 0, 360)
    tropical_indian_ocean_region = (-25, 25, 30, 110)
    northern_atlantic_ocean_region = (0, 15, -60%360, -15%360)

    # Calculate regional biases
    data_so = calculate_regional_bias(data_em, lat, lon, southern_ocean_region)
    data_tio = calculate_regional_bias(data_em, lat, lon, tropical_indian_ocean_region)
    data_nao = calculate_regional_bias(data_em, lat, lon, northern_atlantic_ocean_region)
    data_ct = calculate_regional_bias(data_em, lat, lon, cold_tongue_region)

    plot_cold_tongue_remote_biases_scatter(data_tio, data_nao, data_so, data_ct, filename="SFig14_1.png")

    # ============================================================
    # Define regions of interest
    indo_pacific_warm_pool_region = (-25, 25, 40, -160%360)
    
    # Calculate regional biases
    data_ipwp = calculate_regional_bias(data_em, lat, lon, indo_pacific_warm_pool_region)

    data_sep = np.stack([calc_southeast_pacific_region_average(v) for v in data])
    tdata_sep = np.apply_along_axis(mk_test,1,np.nan_to_num(data_sep))*10
    tdata_sep = np.array([wgt_mean(v,lons,lats) for v in tdata_sep]) 

    plot_pacific_trend_discrepancy_remote_biases_scatter(data_ipwp, data_so, tdata_sep, filename="SFig14_2.png"))