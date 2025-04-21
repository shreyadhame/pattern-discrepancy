  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relationship between mean state and trend discrepancy
"""

__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
## Load modules
import numpy as np
import warnings
from scipy.odr import ODR, Model, RealData
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Suppress warnings
warnings.filterwarnings('ignore')

# Local modules
from import_highresmip_data import *
from help_func import *
from plot_map import *

#============================================================
# Constants
START_YEAR = 1979
END_YEAR = 2014
(LAT1, LAT2) = (-5, 5)
(LON1, LON2) = (180, -95 % 360)
DPI = 750

COLORS = ["#0072B2", "#332288", "#D55E00", "#AA4499", "#009E73", "#56B4E9", "#2f2f2f"]
COLORS_RPT = [color * count for color, count in zip(COLORS, [2, 2, 3, 3, 3, 4, 4])]
MODEL_LABELS = ['MPI-ESM1-2', 'HadGEM3-GC31', 'ECMWF-IFS', 'EC-Earth3P', 'CMCC-CM2', 'CNRM-CM6-1', 'CESM1-CAM5']

#============================================================
def linear_model(B, x):
    """
    Linear model y = B[0] * x + B[1]

    Parameters:
    B (array-like): Parameters [slope, intercept].
    x (array-like): Independent variable values.

    Returns:
    array-like: Dependent variable values.
    """
    return B[0] * x + B[1]

def orthogonal_regression(x, y):
    """
    Perform orthogonal regression using scipy.odr.

    Parameters:
    x (array-like): Independent variable values.
    y (array-like): Dependent variable values.

    Returns:
    slope (float): Slope of the fitted line.
    intercept (float): Intercept of the fitted line.
    """
    # Create a model for the regression
    model = Model(linear_model)

    # Create a RealData object with the input data
    data = RealData(x, y)

    # Create an ODR object
    odrr = ODR(data, model, beta0=[0.2, 1])  # Initial guess for slope and intercept

    # Run the regression
    output = odrr.run()

    # Extract the slope and intercept
    slope, intercept = output.beta

    return slope, intercept

#============================================================
def plot_relationship(mean_ctw, trends_ctw, filename="Relationship.png"):
    """
    Plot the relationship between cold tongue bias and SST trend discrepancy.
    """
    try:
        fig = plt.figure(figsize=(3.74, 3.74))
        
        # Grid specification
        spec = gridspec.GridSpec(ncols=1, nrows=1, width_ratios=[1], height_ratios=[1], hspace=0., wspace=0.)
        ax = fig.add_subplot(spec[0, 0])

        # Data preparation
        x = mean_ctw[:-1] - mean_ctw[-1]
        y = trends_ctw[:-1] - trends_ctw[-1]

        #Marker sizes
        s = np.arange(1,len(COLORS_RPT)+1)
        sizes = [(s[i]+3)**2 for i in range(len(s))][::-1]

        # Scatter plot
        for i in range(len(COLORS_RPT)):
            ax.scatter(x[i], y[i], s=sizes[i], marker='o', lw=1.5, facecolors='none', edgecolors=COLORS_RPT[i])

        # Fit a linear regression line
        ax.set_xlim(-1.5, 0.5)
        ax.set_ylim(-0.01, 0.35)
        
        b, a = orthogonal_regression(x, y)
        xseq = np.linspace(np.nanmin(x), np.nanmax(x), num=100)
        ax.plot(xseq, a + b * xseq, color="#909090", lw=0.4, ls='--')

        # Calculate r and r^2 values
        y_pred = b * x + a
        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        r_value = np.sqrt(r_squared) if r_squared >= 0 else 0
        
        # Display correlation coefficient
        ax.text(0.05, 0.92, f'r = {np.round(r_value, 1)}', transform=ax.transAxes, fontsize=12)

        # Zero lines
        ax.axhline(0., color='#909090', zorder=-1)
        ax.axvline(0., color='#909090', zorder=-1)
        
        # Hide the right and top spines
        ax.spines[['right', 'top']].set_visible(False)
        
        # Labels and title
        ax.set_xlabel('Cold tongue bias [$^\circ$C]')
        ax.set_ylabel('Central-to-eastern equatorial Pacific Ocean\ntrend discrepancy [$^\circ$C $decade^{-1}$]')

        # Add model labels
        x_text = 0.4
        y_text_start = 0.02
        inc = 0.03

        for i, name in enumerate(MODEL_LABELS):
            ax.text(x_text, y_text_start + i * inc, name, fontsize=10, color=colors_model[len(colors_model) - i - 1])
    
        plt.suptitle('Relationship between cold tongue bias and\nSST trend discrepancy (1979 - 2014)', fontsize=11)
    
        plt.savefig(filename, dpi=DPI, bbox_inches="tight")
        print(f"Relationship plot saved to {filename}")

    except Exception as e:
        print(f"Error in plot_relationship: {e}")
    
def main():
    #All models 
    data = [
        nhr_tos, 
        mer_tos, 
        ehr_tos, 
        echr_tos, 
        emr_tos, 
        hm_tos, 
        cnhr_tos, 
        mm_tos, 
        mhr_tos, 
        elr_tos, 
        cmsr_tos, 
        eclr_tos, 
        nle_tos, 
        cnlr_tos, 
        ll_tos, 
        mlr_tos, 
        ersst_tos]
    
    # Drift corrected data
    data_dc = [
        nhr_tos_dc, 
        mer_tos_dc, 
        ehr_tos_dc, 
        echr_tos_dc, 
        emr_tos_dc, 
        hm_tos_dc, 
        cnhr_tos_dc, 
        mm_tos_dc, 
        mhr_tos_dc, 
        elr_tos_dc, 
        cmsr_tos_dc, 
        eclr_tos, 
        nle_tos, 
        cnlr_tos_dc, 
        ll_tos_dc, 
        mlr_tos_dc, 
        ersst_tos]

    # Take Ensemble mean
    data_em = [v.mean(axis=0) for v in data]
    data_dc = [v.mean(axis=0) for v in data_dc]

    # Select time 
    years = mlr_tos.time.dt.year
    data_em = [seltime(v, years, START_YEAR, END_YEAR) for v in data_em]
    data_dc = [seltime(v, years, START_YEAR, END_YEAR) for v in data_dc]

    # Calculate mean and trends for cold tongue region
    lats = lat.sel(lat=slice(LAT1, LAT2))
    lons = lon.sel(lon=slice(LON1, LON2))
    mean_ct = np.stack([v.sel(lat=slice(LAT1,LAT2),lon=slice(LON1,LON2)) for v in data_em])
    mean_ctw = np.array([wgt_mean(v,lons,lats) for v in mean_ct])

    data_dc_ct = np.stack([v.sel(lat=slice(LAT1,LAT2),lon=slice(LON1,LON2)) for v in data_dc])
    #Calculate trends 
    trends_ct = np.apply_along_axis(mk_test,1,np.nan_to_num(data_dc_ct))*10
    trends_ctw = np.array([wgt_mean(v,lons,lats) for v in trends_ct])

    # Plot
    plot_relationship(mean_ctw, trends_ctw, filename="Fig5.png")

# ============================================================
## Execute script
if __name__ == "__main__":
    main()