Code for the analysis and figures in Dhame, S., Olonscheck, D., Rugenstein, M. (2025). Higher Resolution Climate Models Do Not Consistently Reproduce the Observed Tropical Pacific Warming Pattern. Journal of Climate.

The ERSSTv5 and COBE datasets were downloaded from https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html and https://psl.noaa.gov/data/gridded/data.cobe2.html, respectively. ERA5 monthly averaged data for 10m zonal wind was downloaded from https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview. The EN version 4 subsurface ocean temperature data was obtained from the UK Met Office Hadley Centre (https://www.metoffice.gov.uk/hadobs/en4/). The HighResMIP model datasets are available at the Earth System Grid Federation (ESGF) at https://esgf-data.dkrz.de/projects/esgf-dkrz/. The high resolution CESM1.3 model datasets are available at https://ihesp.github.io/archive/products/ds_archive/Sunway_Runs.html. Low-frequency component analysis was conducted using code available at https://github.com/karenamckinnon/forcesmip/blob/main/notebooks/ForceSMIP_LFCA.ipynb. Causal analysis was conducted using code available at https://github.com/jakobrunge/tigramite. 

Method:

    All datasets were interpolated onto 2.5 x 2.5 global grid using the bilinear interpolation method of Climate Data Operator (CDO)
    Run individual scripts for figures
