#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model comparison of regional SST trend discrepancies.
"""

__title__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

# ============================================================
import itertools
import warnings
warnings.filterwarnings('ignore')

# Local modules
from import_highresmip_data import *
from help_func import *
from plot_map import *

# ============================================================
#Constants 
START_YEAR = 1979
END_YEAR = 2014

COLORS = ["#0072B2", "#332288", "#D55E00", "#AA4499", "#009E73", "#56B4E9", "#2f2f2f"]
COLORS_RPT = [color * count for color, count in zip(COLORS, [2, 2, 3, 3, 3, 4, 4])]
COLORS_MODEL = ["#0072B2", "#332288", "#D55E00", "#AA4499", "#009E73", "#56B4E9", "#2f2f2f"]
MODEL_LABELS = ['MPI-ESM1-2', 'HadGEM3-GC31', 'ECMWF-IFS', 'EC-Earth3P', 'CMCC-CM2', 'CNRM-CM6-1', 'CESM1-CAM5']

DPI = 750

# Region Definitions
LAT_MIN = -5
LAT_MAX = 5
EP_LON_MIN = 180%360
EP_LON_MAX = -95%360

WP_LON_MIN = 110%360
WP_LON_MAX = 180%360

# ============================================================
def plot_discrepancy(tdata, lat, lon, mlabels, colors, markers, filename="Discrepancy.png"):
    """Plots regional trend discrepancies for different models."""
    try:
        fig, ax = plt.subplots(figsize=(7.48, 2.5))
        pplt.rc.update({"xtick.minor.visible": False})

        yloc = np.arange(len(tdata))[::-1]
        y = np.repeat(yloc, [len(d) for d in tdata])
        c = np.repeat(colors, [len(d) for d in tdata])
        m = np.repeat(markers[:len(tdata) - 1], [len(d) for d in tdata[:-1]]).squeeze()

        tp = tdata[:-1] - tdata[-1]

        for i in range(len(tdata[:-1])):  # for each of the features
            mi = m[i]  # marker for ith feature
            yi = y[i]  # y array for ith feature
            ci = c[i]  # color for ith feature
            ax1.scatter(yi, tp[i], marker=mi, color=ci, s=45, facecolors="none")

        ax.set_xticklabels(mlabels[::-1], fontsize=9)
        ax.set_title("(a) Original data", fontsize=12)
        ax.set_ylabel('$^\circ$C $decade^{-1}$', size=10)
        ax.set_xticks(np.arange(len(tdata)))

        lines = [4.5, 8.5, 11.5, 14.5, 17.5, 19.5, 21.5]
        ax.set_xlim(0.5, len(tdata) - 0.5)
        ax.invert_xaxis()
        ax.axvline(lines, linestyle="--", color="k", lw=0.5, alpha=0.8)
        ax.axhline(0.0, color="k", lw=0.5, zorder=-1)
        ax.axvspan([0.5, 1.5, 4.5, 5.5, 6.5, 8.5, 9.5, 11.5, 12.5, 14.5, 15.5, 17.5, 19.5],
                   [1.5, 2.5, 5.5, 6.5, 7.5, 9.5, 10.5, 12.5, 13.5, 15.5, 16.5, 18.5, 20.5],
                   facecolor="k", alpha=0.1)

        y = -0.32
        for line, name, color in zip(lines, MODEL_LABELS, COLORS_MODEL[::-1]):
            ax.text(line - (2 if name in ["MPI-ESM1-2", "HadGEM3-GC31"] else 1.5), y, name,
                    fontsize=9, color=color, ha="center", va="center")

        plt.savefig(filename, dpi=DPI, bbox_inches="tight")
        print(f"Discrepancy plot saved to {filename}")

    except Exception as e:
        print(f"Error in plot_discrepancy: {e}")

def main():
    """Main function for plotting."""
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
        ersst_tos,
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

    # Select time
    time = mlr_tos.time
    reftime = time.sel(time=slice(str(START_YEAR), str(END_YEAR)))
    years = np.arange(1950,2014)
    ser = np.arange(len(reftime))
    mdata = [seltime(v, years, START_YEAR, END_YEAR) for v in data[:-1]]
    data = mdata + [ersst_tos]

    # Define regions of interest
    # East-West Pacific gradient 
    tdata_ep = calc_box_region_trend(LAT_MIN, LAT_MAX, EP_LON_MIN, EP_LON_MAX, data)
    tdata_wp = calc_box_region_trend(LAT_MIN, LAT_MAX, WP_LON_MIN, WP_LON_MAX, data)
    tdata_ew = tdata_ep - tdata_wp

    nmod = len(data) - 1
    colors = list(itertools.chain.from_iterable(COLORS_RPT))
    markers = ['o']*(nmod+1)

    # Plot discrepancy
    plot_discrepancy(tdata_ew, lon, lat, mlabels, COLORS_RPT, markers, filename='Fig4.png')

# ============================================================
## Execute script
if __name__ == "__main__":
    main()