#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__title__ = "Helper functions"
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

# General modules
import numpy as np
import pymannkendall as mk
from scipy import stats

#============================================================
def seltime(var,years,year1,year2):
    """ Select time period of analysis """
    ind_start=int(np.abs(years-(year1)).argmin())
    ind_end=int(np.abs(years-(year2)).argmin())+1
    vart = var[:,ind_start:ind_end]
    return vart

def seltime_xr(var, years, start_year, end_year):
    return var.sel(time=slice(str(start_year), str(end_year)))

def selreg(var, lon, lat, lon1, lon2, lat1, lat2):
    """
    Selects a subregion from a variable based on latitude and longitude bounds.
    """
    # Find index ranges for latitude and longitude
    ind_start_lat = int(np.abs(lat - lat1).argmin())
    ind_end_lat = int(np.abs(lat - lat2).argmin()) + 1
    ind_start_lon = int(np.abs(lon - lon1).argmin())
    ind_end_lon = int(np.abs(lon - lon2).argmin()) + 1

    # Extract longitude and latitude arrays for the selected region
    lons = lon[ind_start_lon:ind_end_lon]
    lats = lat[ind_start_lat:ind_end_lat]

    # Extract the subregion from `var` based on its number of dimensions
    if var.ndim == 3:
        box = var[:, ind_start_lat:ind_end_lat, ind_start_lon:ind_end_lon]
    elif var.ndim == 4:
        box = var[:, :, ind_start_lat:ind_end_lat, ind_start_lon:ind_end_lon]
    elif var.ndim == 5:
        box = var[:, :, :, ind_start_lat:ind_end_lat, ind_start_lon:ind_end_lon]
    else:
        raise ValueError("Input variable `var` must have 3, 4, or 5 dimensions.")

    return box, lons, lats

def wgt_mean(var: xr.DataArray, lon: xr.DataArray, lat: xr.DataArray) -> np.float64:
    """
    Calculate the weighted mean of a variable, considering latitude.

    Args:
        var (xr.DataArray): The variable to average (time, lat, lon).
        lon (xr.DataArray): Longitude coordinates.
        lat (xr.DataArray): Latitude coordinates.

    Returns:
        np.float64: The weighted mean of the variable.
    """
    var_ma = ma.masked_invalid(var)
    wgtmat = np.cos(np.tile(abs(lat.values[:,None])*np.pi/180,(1,len(lon))))[np.newaxis,...]
    var_mean = np.ma.sum((var_ma*wgtmat*~var_ma.mask))/(np.ma.sum(wgtmat * ~var_ma.mask))
    return var_mean

def lag1_acf(x: np.ndarray, nlags: int = 1) -> np.ndarray:
    """
    Calculate the Lag-1 Autocorrelation Function (ACF).

    Args:
        x (np.ndarray): A 1D numpy array of data.
        nlags (int): The number of lags to calculate.  Defaults to 1.

    Returns:
        np.ndarray: The Lag-1 autocorrelation coefficient.
    """
    y = x - np.nanmean(x)
    n = len(x)
    d = n * np.ones(2 * n - 1)

    acov = (np.correlate(y, y, 'full') / d)[n - 1:]
    acf = acov[:nlags] / acov[0]
    return acf

def mk_test(x, alpha=0.10):
    """Perform Mann-Kendall test for trend."""
    acf = lag1_acf(x)[0]
    n = len(x)
    r1 = (-1 + 1.96 * np.sqrt(n - 2)) / (n - 1)
    r2 = (-1 - 1.96 * np.sqrt(n - 2)) / (n - 1)
    
    if (acf > 0 and acf > r1) or (acf < 0 and acf < r2):
        return mk.yue_wang_modification_test(x).slope
    else:
        return mk.original_test(x).slope

def calc_sig(x: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate the p-value for trend significance using the Mann-Kendall test.

    Applies the Mann-Kendall test for trend significance, considering 
    serial autocorrelation.

    Args:
        x (np.ndarray): A 1D numpy array of data.
        alpha (float): The significance level (p-value threshold). Defaults to 0.05.

    Returns:
        float: The p-value of the significance test.
    """
    acf = lag1_acf(x)

    r1 = (-1 + 1.96*np.sqrt(len(x)-2))/len(x)-1
    r2 = (-1 - 1.96*np.sqrt(len(x)-2))/len(x)-1

    if (acf > 0) and (acf > r1):
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.yue_wang_modification_test(x)
    elif (acf < 0) and (acf < r2):
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.yue_wang_modification_test(x)
    else:
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(x)
    return p

def calc_linregress(series, var):
    """Calculate linear regression for 2D data."""
    results = stats.linregress(series[:, np.newaxis, np.newaxis], var)
    return results.slope, results.intercept, results.rvalue, results.pvalue, results.stderr

def calc_southeast_pacific_region_average(var):

    lat_ends = np.repeat(np.arange(-46.25, -1.25, 2.5), 2)[::-1]
    lon_ends = np.arange(-160 % 360, -70 % 360, 2.5)
    
    lat_slices = slice(lat_ends[0], -5)
    lon_slices = [slice(lon_ends[i], lon_ends[i+1]) for i in range(len(lon_ends)-1)]

    boxes = [var.sel(lat=lat_slices, lon=lon_slice) for lon_slice in lon_slices]

    boxm = [xr.apply_ufunc(
        wgt_mean,
        box,
        input_core_dims=[['lat', 'lon']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    ) for box in boxes]

    # Take mean 
    var_reg = np.nanmean(np.stack(boxm), axis=0)

    return var_reg

def calc_southeast_pacific_region_trend(data):
    # Calculate SEP average for individual ensemble members
    data_sep = []
    for i in range(len(data)):
        data_sep.append([calc_southeast_pacific_region_average(data[i][j]) for j in range(len(data[i]))])
    data_sep = np.concatenate(data_sep)

    # Calculate trend
    tdata_sep = np.apply_along_axis(mk_test,1,np.nan_to_num(data_sep))*10

    return tdata_sep

def calc_box_region_trend(lat_min, lat_max, lon_min, lon_max, datam):
    """
    Weighted mean results for the Eastern Pacific bias.
    """
    # Select latitude and longitude slices
    latitude_slice = lat.sel(lat=slice(lat_min, lat_max))
    longitude_slice = lon.sel(lon=slice(lon_min, lon_max))

    # Extract data for the specified region
    regional_data = [var.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        for var in datam]

    # Concatenate data along a new axis
    concatenated_data = np.concatenate(regional_data)

    # Apply Mann-Kendall test and scale results by a factor of 10 (decadal trend)
    mann_kendall_results = [np.apply_along_axis(mk_test, axis=0, arr=np.nan_to_num(data)) * 10
    for data in concatenated_data]

    # Compute weighted mean
    reg_trend = np.array([wgt_mean(result, longitude_slice, latitude_slice) for result in mann_kendall_results])

    return reg_trend

def calc_d20(theta,depth):
    """
    Calculates depth of the thermocline (maximum gradient change)
    """
    # Calculate the temperature gradient
    gradient = np.gradient(theta, depth)
    # Find the index of the maximum absolute gradient
    max_gradient_index = np.argmax(np.abs(gradient))
    # Get the depth of the maximum gradient (thermocline)
    thermocline_depth = depth[max_gradient_index]
    return thermocline_depth

def calculate_thermocline_tilt(thermocline_depth, lons):
    """
    Calculate thermocline tilt based on depth differences across longitude.
    """
    western_depth = thermocline_depth[:, 0].mean(axis=0)
    eastern_depth = thermocline_depth[:, -1].mean(axis=0)
    
    distance = (lons[-1] - lons[0]) * 111  # Approximate km per degree
    tilt = (eastern_depth - western_depth) / distance
    
    return tilt

def compute_trend_and_significance(data, axis):
    """
    Compute the Mann-Kendall trend and significance for the given data.
    """
    trend = np.apply_along_axis(mk_test, axis, data) * 10
    pval = np.apply_along_axis(calc_sig, axis, data)
    sig_mask = mask_significance(pval)
    return trend, sig_mask

def calculate_trends(data, time, start_year, end_year, min_len=30, scale=10):
    """Calculate trends for periods longer than 30 years using Mann-Kendall test"""
    trends = []
    for i in range(start_year, end_year):
        for j in range(start_year, end_year):
            ind1 = np.where(time.dt.year == i)[0][0]
            ind2 = np.where(time.dt.year == j)[0][0]
            chunk = data[ind1:ind2]
            if len(chunk) >= min_len:
                trends.append(mk_test(chunk) * scale)
            else:
                trends.append(np.nan)
    return trends