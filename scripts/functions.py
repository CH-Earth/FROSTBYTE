###########################################################################################################
# This Python script contains all the Python functions needed to run the data-driven forecasting workflow #
# Note: The functions are ordered alphabetically and separated by ###                                     #
###########################################################################################################

# Import required modules
import datetime
from datetime import timedelta, date
import geopandas as gpd
import itertools
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import properscoring as ps
import random
import rasterio
from rasterio.plot import show
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import norm, circmean
import seaborn as sns
from shapely.geometry import Point
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import sys
from textwrap import wrap
import warnings
import xarray as xr

def artificial_gap_filling(original_data, iterations, artificial_gap_perc, window_days, min_obs_corr, min_obs_cdf, min_corr, min_obs_KGE, flag):

    """Creating random artificial gaps in the original dataset for each month & station, and running the gap filling function to assess its performance.

    Keyword arguments:
    ------------------
    - original_data: Pandas DataFrame of original stations' observations dataset, to which data will be removed for artificial gap filling
    - iterations: Positive integer denoting the number of times we want to repeat the artificial gap filling (we remove data at random each time in the original dataset)
    - artificial_gap_perc: Percentage between 1 and 100 for the amount of data to remove for each station & month's first day
    - window_days: Positive integer denoting the number of days to select data for around a certain doy, to calculate correlations
    - min_obs_corr: Positive integer for the minimum number of overlapping observations required to calculate the correlation between 2 stations
    - min_obs_cdf: Positive integer for the minimum number of stations required to calculate a station's cdf
    - min_corr: Value between 0 and 1 for the minimum correlation value required to keep a donor station
    - min_obs_KGE: Positive integer for the minimum number of stations required to calculate a station's cdf
    - flag: Integer to plot the gap filled values vs the observed values (1) or not (0)

    Returns:
    --------
    - evaluation: Dictionary containing the artificial gap filling evaluation results for several metrics for each month's first day, station & iteration
    - fig (optional): A figure of the gap filled vs. the actual SWE observations for each first day of the month

    """

    # Set up the figure
    if flag == 1:
        ncols = 3
        fig, axs = plt.subplots(4, ncols, sharex=False, sharey=False, figsize=(8,10))
        plot_col = -1
        row = 0

    # Identify stations for gap filling (without P & external SWE stations (buffer) as we don't do any gap filling for these)
    cols = [c for c in original_data.columns if 'precip' not in c and 'ext' not in c]

    # Create an empty dictionary to store the metric values for each month, station & iteration
    evaluation = {}
    metrics = ['RMSE', "KGE''", "KGE''_corr", "KGE''_bias", "KGE''_var"]
    for m in metrics:
        evaluation[m] = np.ones((12, len(cols), iterations)) * np.nan

    # Calculate correlations between stations that have overlapping observations
    corr = calculate_stations_doy_corr(original_data, window_days, min_obs_corr)

    # loop over months
    for mo in range(1,12+1):

        print('Working on month '+str(mo))

        # controls for plotting on right subplot
        if flag == 1:
            plot_col += 1
            if plot_col == ncols:
                row += 1
                plot_col = 0

        # loop over iterations
        for i in range(iterations):

            # initialize counter to assign results to the right station
            elem = -1

            # looping over stations
            for s in cols:

                # update counter to assign results to the right station
                elem += 1

                # duplicate original data to create artificial gaps from this
                artificial_gaps_data = original_data.copy()

                # remove all missing values for a given station for which to perform gap filling
                station_nomissing_values = pd.DataFrame(artificial_gaps_data[s].dropna())

                # add DOY to select data to gap fill within a time window around first day of month
                station_nomissing_values['doy'] = station_nomissing_values.index.dayofyear

                # calculate the doy corresponding to the date - using 2010 as common year (not leap year)
                doy = int(datetime.datetime(2010,mo,1).strftime('%j'))

                # calculate the start & end doys of the time window for quantile mapping, with caution around the start & end of the calendar year
                window_startdoy = (doy-window_days)%365
                window_startdoy = 365 if window_startdoy == 0 else window_startdoy
                window_enddoy = (doy+window_days)%365
                window_enddoy = 366 if window_enddoy == 0 or window_enddoy == 365 else window_enddoy

                # select data within time window
                if window_startdoy > window_enddoy:
                    data_window = station_nomissing_values[(station_nomissing_values['doy']>=window_startdoy) | (station_nomissing_values['doy'] <= window_enddoy)]
                else:
                    data_window = station_nomissing_values[(station_nomissing_values['doy']>=window_startdoy) & (station_nomissing_values['doy'] <= window_enddoy)]

                # Select target data within this time window
                data_window_target = data_window[s]

                # calculate the number of observations to remove for this station & month's first day
                n = int(len(data_window.index) * artificial_gap_perc / 100)

                # if the number of observations is above zero we can proceed with the gap filling
                if n > 0:

                    # randomly select n dates from the station's data (no duplicates) and remove them from the original dataset - if 100% is removed then all dates will be selected
                    if artificial_gap_perc == 100:
                        dates_to_remove = data_window.index
                    else:
                        dates_to_remove = data_window.index[random.sample(range(0, len(data_window.index)), n)]
                    artificial_gaps_data[s].loc[dates_to_remove] = np.nan
                    artificial_gaps_data = artificial_gaps_data.loc[dates_to_remove]

                    # Keep only SWE station to gap fill
                    gapfilled_data = artificial_gaps_data[s].copy()

                    # Identify dates for gap filling
                    time_index = data_window.dropna().index

                    # Loop over dates for gap filling
                    for d in time_index:

                        # Get the doy corresponding to the date
                        doy = data_window.dropna().loc[d,'doy']

                        # Get IDs of all stations with data for this date (and within time window)
                        data_window_allstations = artificial_gaps_data.dropna(axis=1, how='all')
                        non_missing_stations = [c for c in data_window_allstations.columns]
                        data_window_allstations['days_to_date'] = abs((d - data_window_allstations.index).days)

                        # We can continue if there are enough target data to build cdf
                        if len(data_window_target.index) >= min_obs_cdf:

                            # Get ids of all stations with correlations >= a minimum correlation for this doy, not including the target station itself
                            non_missing_corr = corr[doy][s].dropna()
                            non_missing_corr = non_missing_corr[non_missing_corr.index.isin(non_missing_stations)]
                            potential_donor_stations = non_missing_corr[non_missing_corr >= min_corr].index.values
                            potential_donor_stations = [c for c in potential_donor_stations if s not in c]

                            # If there is at least one potential donor station, proceed
                            if len(potential_donor_stations) > 0:

                                # Sort the donor stations from highest to lowest value
                                potential_donor_stations_sorted = corr[doy].loc[potential_donor_stations,s].dropna().sort_values(ascending=False).index.values

                                # Loop over sorted donor stations until I find one with enough data to build a cdf
                                for donor_station in potential_donor_stations_sorted:

                                    # Select data within time window for this doy from all years
                                    data_window_donor = data_window_allstations[donor_station].dropna()

                                    # We can continue if there are enough donor data to build cdf
                                    if len(data_window_donor.index) >= min_obs_cdf:

                                        # If the donor station has multiple values within the window, we keep the closest donor station value to the date we are gap filling
                                        sorted_data_window = data_window_allstations.sort_values(by=['days_to_date'])
                                        value_donor = sorted_data_window[donor_station].dropna()[0]

                                        # Perform the gap filling using quantile mapping
                                        value_target = quantile_mapping(data_window_donor, data_window_target, value_donor, min_obs_cdf, flag=0)

                                        if value_target != None:
                                            gapfilled_data.loc[d] = value_target

                                        break

                                    else:
                                        continue

                    # combine observed & predicted data into a single Pandas dataframe
                    # results = gapfilled_data[s].loc[dates_to_remove]
                    results = gapfilled_data.to_frame(name='pre')
                    results['obs'] = original_data[s].loc[dates_to_remove]
                    results = results.dropna()

                    # plot the gap filled vs the observed values
                    if flag == 1:
                        axs[row,plot_col].scatter(results['obs'], results['pre'], color='b', alpha=.3)
                        axs[row,plot_col].set_title('month'+str(mo))
                        if row == 3 and plot_col == 0:
                            axs[row,plot_col].set_xlabel('observed')
                            axs[row,plot_col].set_ylabel('infilling')

                    # if there are no predicted values set the metrics to nan
                    if results.empty == True:
                        for m in metrics:
                            evaluation[m][mo-1,elem,i] = np.nan

                    # otherwise proceed with evaluating the gap filling performance
                    else:
                        rmse = mean_squared_error(results['obs'], results['pre'], squared=False)
                        kge_prime_prime = KGE_C2021(results['obs'].values, results['pre'].values, min_obs_KGE)
                        evaluation['RMSE'][mo-1,elem,i] = rmse
                        evaluation["KGE''"][mo-1,elem,i] = kge_prime_prime['KGE']
                        evaluation["KGE''_corr"][mo-1,elem,i] = kge_prime_prime['r']
                        evaluation["KGE''_bias"][mo-1,elem,i] = kge_prime_prime['beta']
                        evaluation["KGE''_var"][mo-1,elem,i] = kge_prime_prime['alpha']

                # else if the number of observations is zero we go to the next station
                else:
                    continue

    if flag == 1:
        plt.tight_layout()
        return evaluation, fig

    else:
        return evaluation

###

def basins_maps(basins, method, variable, nival_start_doy, nival_end_doy, domain):

    """Plots two maps of basins provided, one that shows the basins' shapes & one that shows the basins' outlets.

    Keyword arguments:
    ------------------
    - basins: Pandas GeoDataFrame of all basin shapefiles available to subset from
    - method: String of the metric used to identify streamflow peaks (shown on maps' title)
    - variable: String of the column label to be used for colouring the maps
    - nival_start_doy: Integer day of year (doy) of the start of the nival period (default=pre-defined at the top of the Notebook)
    - nival_end_doy: Integer day of year (doy) of the end of the nival period (default=pre-defined at the top of the Notebook)
    - domain: String of the geographical domain to plot

    Returns:
    --------
    - Two maps of basins.

    """

    # fig, axs = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(20,10))

    # # Plot map of basins' outlines
    # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # world_subdomain = world[world['name']==domain].copy()
    # world_NA = world[world['continent']=='North America'].copy() # this will have to be modified, I can't get the world map plot to work so need to subselect continent to plot
    # world_NA_albers = world_NA.to_crs("ESRI:102008")
    # world_subdomain_albers = world_subdomain.to_crs("ESRI:102008")
    # world_NA_albers.plot(ax=axs[0], linewidth=1, edgecolor='grey', color='white')
    # basins_albers = basins.to_crs("ESRI:102008")
    # basins_albers.sort_values(by=['Shp_Area'], ascending=False).plot(ax=axs[0], column=variable, cmap=plt.cm.viridis, vmin=nival_start_doy, vmax=nival_end_doy, legend=True, legend_kwds={'label':'DOY'})  # sorting the basins by area so that smaller basins aren't hidden under larger basins that encompass them
    # minx, miny, maxx, maxy = np.nanmin(world_subdomain_albers.geometry.bounds.minx),np.nanmin(world_subdomain_albers.geometry.bounds.miny),np.nanmax(world_subdomain_albers.geometry.bounds.maxx),np.nanmax(world_subdomain_albers.geometry.bounds.maxy)
    # axs[0].set_xlim(minx, maxx)
    # axs[0].set_ylim(miny, maxy)
    # axs[0].margins(0)
    # axs[0].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    # axs[0].title.set_text('Nival basins ('+method+')'+title_end)
    # axs[0].text(minx+100,miny+100,'Total basins='+str(len(basins.index)))
    # axs[0].set_facecolor('azure');

    # # Plot map of basins' outlets
    # basins_pt = basins.copy()
    # basins_pt = gpd.GeoDataFrame(basins_pt, geometry=gpd.points_from_xy(basins_pt.outlet_lon, basins_pt.outlet_lat))
    # basins_pt.crs = {"init":"epsg:4326"}
    # basins_pt_albers = basins_pt.to_crs("ESRI:102008")
    # world_NA_albers.plot(ax=axs[1], linewidth=1, edgecolor='grey', color='white')
    # # sc = plt.scatter(basins_pt_albers.outlet_lon.values, basins_pt_albers.outlet_lat.values, c=basins_pt_albers[variable].values, cmap=plt.cm.viridis, vmin=nival_start_doy, vmax=nival_end_doy, alpha=.5)
    # # plt.colorbar(sc, label='DOY')
    # basins_pt_albers.plot(ax=axs[1], column=variable, cmap=plt.cm.viridis, vmin=nival_start_doy, vmax=nival_end_doy, legend=True, legend_kwds={'label':'DOY'}, alpha=.5, markersize=5)
    # axs[1].set_xlim(minx-2, maxx+2)
    # axs[1].set_ylim(miny-2, maxy+2)
    # axs[1].margins(0)
    # axs[1].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    # axs[1].title.set_text('Nival basins outlets ('+method+')'+title_end)
    # axs[1].text(minx+100,miny+100,'Total basins='+str(len(basins.index)))
    # axs[1].set_facecolor('azure');

    # Plot map of basins' outlets
    fig, ax = plt.subplots()
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    continental_subdomain = world[world['continent']==domain]
    continental_subdomain_albers = continental_subdomain.to_crs("ESRI:102008")
    continental_subdomain_albers.plot(ax=ax, figsize=(20,5), linewidth=1, edgecolor='grey', color='white')
    basins_albers = basins.to_crs("ESRI:102008")
    basins_albers.plot(ax=ax, column=variable, cmap=plt.cm.viridis, vmin=nival_start_doy, vmax=nival_end_doy, legend=True, legend_kwds={'label':'Mean peak DOY'}, alpha=.5, markersize=5)
    minx, miny, maxx, maxy = np.nanmin(continental_subdomain_albers.geometry.bounds.minx),np.nanmin(continental_subdomain_albers.geometry.bounds.miny),np.nanmax(continental_subdomain_albers.geometry.bounds.maxx),np.nanmax(continental_subdomain_albers.geometry.bounds.maxy)
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    plt.margins(0)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    title = ax.set_title("\n".join(wrap('Nival basins outlets ('+method+')', 50)))
    plt.text(.01, .01, 'Total basins='+str(len(basins.index)), ha='left', va='bottom', transform=ax.transAxes)
    ax.set_facecolor('azure')
    plt.tight_layout()
    title.set_y(1.05);

    return fig

###

def calculate_stations_doy_corr(stations_obs, window_days, min_obs_corr):

    """Calculating stations' correlations for each day of the year (doy; with a X-day window centered around the doy).

    Keyword arguments:
    ------------------
    - stations_obs: Pandas DataFrame of all SWE & P stations observations
    - window_days: Positive integer denoting the number of days to select data for around a certain doy, to calculate correlations
    - min_obs_corr: Positive integer for the minimum number of overlapping observations required to calculate the correlation between 2 stations

    Returns:
    --------
    - stations_doy_corr: Dictionary containing a Pandas DataFrame of stations correlations for each day of year

    """

    # Set up the dictionary to save all correlations
    stations_doy_corr = {}

    # Duplicate the stations observations Pandas DataFrame and add doy column
    stations_obs_doy = stations_obs.copy()
    stations_obs_doy['doy'] = stations_obs_doy.index.dayofyear

    # Loop over days of the year
    for doy in range(1,366+1):

        # calculate the start & end of the data selection window, with caution around the start & end of the calendar year
        window_start = (doy-window_days)%366
        window_start = 366 if window_start == 0 else window_start
        window_end = (doy+window_days)%366
        window_end = 366 if window_end == 0 else window_end

        # select data for the window of interest
        if window_start > window_end:
            data_window = stations_obs_doy[(stations_obs_doy['doy']>=window_start) | (stations_obs_doy['doy'] <= window_end)]
        else:
            data_window = stations_obs_doy[(stations_obs_doy['doy']>=window_start) & (stations_obs_doy['doy'] <= window_end)]

        # calculate the Pearson product-moment correlations between stations if the minimum number of observations criterium is met
        data_window = data_window.drop(columns=['doy'])
        corr = data_window.corr(method='spearman', min_periods=min_obs_corr)
        # np.fill_diagonal(corr.values, np.nan)

        # save correlation for the doy to the dictionary
        stations_doy_corr[doy] = corr

    return stations_doy_corr

###

def circular_stats(doy, year_length):

    """Calculates circular statistics. See: https://onlinelibrary.wiley.com/doi/abs/10.1002/hyp.7625

    Keyword arguments:
    ------------------
    - doy: Numpy array of day of year (doy) values for identified streamflow peaks
    - year_length: Numpy array of year length values for identified streamflow peaks

    Returns:
    --------
    - theta_rad: Numpy array of the angular values (in radians) that correspond to the doy
    - regularity: Dimensionless value indicating the spread of the data (ranges from 0: high spread to 1: low spread)

    """

    theta_rad = doy * ((2*math.pi)/year_length)
    x_coord = np.mean([math.cos(x) for x in theta_rad])
    y_coord = np.mean([math.sin(x) for x in theta_rad])
    regularity = math.sqrt(x_coord**2 + y_coord**2)

    return theta_rad, regularity

###

def continuous_rank_prob_score(Qobs, Qfc_ens, min_obs):

    """Calculates the Continuous Rank Probability Score (CRPS) and the Continuous Rank Probability Skill Score (CRPSS).
    The CRPS is a measure of the difference between the predicted (from an ensemble or probabilistic forecast) and the observed cumulative distribution functions (cdf).
    The CRPSS measures the performance (CRPS) of the forecast against a baseline (e.g., the observation climatology).
    CRPS range: 0 to +Inf. Perfect score: 0. Units: Same as variable measured.
    CRPSS range: -Inf to 1. Perfect score: 1. Units: Unitless.
    Characteristics: It is equivalent to the mean absolute error (MAE) for deterministic forecasts.
    For more info, see the Python CRPS package documentation: https://pypi.org/project/properscoring/

    Keyword arguments:
    ------------------
    - Qobs: xarray DataArray containing a timeseries of observed flow values.
    - Qfc_ens: xarray DataArray containing a timeseries of ensemble flow forecasts.
    - min_obs: Positive integer for the number of minimum number of observation-hindcast pairs required to calculate the various metrics

    Returns:
    --------
    - CRPS: Float of the CRPS value between the ensemble forecasts & observations.
    - CRPSS: Float of the CRPSS value between the ensemble forecasts & observations.

    """

    # Construct a baseline from the observations climatology - removing the obs each year
    baseline = np.array([Qobs]*len(Qobs))
    np.fill_diagonal(baseline, np.nan)

    if len(Qobs) >= min_obs:

        # Calculate CRPS and CRPSS
        CRPS = ps.crps_ensemble(Qobs, Qfc_ens).mean()
        CRPS_baseline = ps.crps_ensemble(Qobs, baseline).mean()
        CRPSS = 1 - CRPS / CRPS_baseline

    else:

        CRPS, CRPSS = np.nan, np.nan

    return CRPS, CRPSS

###

def corr_coeff_squared(Qobs, Qfc_det, min_obs):

    """Calculates the squared correlation coefficient between deterministic forecasts & observations.
    Answers the question: How well did the forecast values correspond to the observed values?
    Range: -1 to 1. Perfect score: 1. Units: Unitless.
    Characteristics: Good measure of linear association - visually, the correlation measures how close the points of a scatter plot are to a straight line. Does not take forecast bias into account. Sensitive to outliers.

    Keyword arguments:
    ------------------
    - Qobs: Numpy Array containing a timeseries of observed flow values.
    - Qfc_det: Numpy Array containing a timeseries of deterministic flow forecasts (e.g., medians or means of the ensemble forecasts).
    - min_obs: Positive integer for the number of minimum number of observation-hindcast pairs required to calculate the various metrics.

    Returns:
    --------
    - r_squared: Float of the squared correlation coefficient between the deterministic forecasts & observations.

    """

    ind_nan = np.isnan(Qobs) | np.isnan(Qfc_det)
    obs = Qobs[~ind_nan]
    pre = Qfc_det[~ind_nan]

    if len(obs) >= min_obs:

        # Check to see if all forecast values are the same. If they are r cannot be calculated and is set to 0
        # For more info: https://hess.copernicus.org/articles/23/4323/2019/ (Section 2)
        if np.all(pre == pre[0]) == True:

            r_squared = 0

        else:

            r_squared = stats.pearsonr(obs, pre)[0] ** 2

    else:

        r_squared = np.nan

    return r_squared

###

def cumulative_hydrographs(basins, streamflow_obs, month_start_water_year, day_start_water_year):

    """Plots normalized cumulative climatological streamflows for the provided basins, differenciating between nival and glacial basins.

    Keyword arguments:
    ------------------
    - basins: Pandas GeoDataFrame of all basins to plot
    - streamflow_obs: xarray Dataset of streamflow observations
    - month_start_water_year: Integer of the water year starting month
    - day_start_water_year: Integer of the water year starting day of the month

    Returns:
    --------
    - A plot of all basins' cumulative climatological hydrographs.

    """

    # Set up figure
    fig = plt.figure()

    # We expect to see RuntimeWarnings in this block due to missing values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        cumsum_norm_nivalbasins = np.ones((len(basins[basins.regime=='nival'].index),366))
        cumsum_norm_glacialbasins = np.ones((len(basins[basins.regime=='glacial'].index),366))

        nniv = -1
        nglac = -1

        # calculate the number of days between the water year and calendar start
        start_water_year = date(2020,month_start_water_year,day_start_water_year)
        start_calendar_year = date(2021,1,1)
        delta = (start_calendar_year - start_water_year).days

        # extract all stations for plotting
        stations_to_plot = [x for x in basins.index.values]

        # loop over the stations
        for s in stations_to_plot:

            # read streamflow observations for basin as xarray DataArray
            streamflow_data_da = streamflow_obs.Flow.sel(Station_ID=s)

            # calculate the climatological mean flow for each DOY
            doy_mean = streamflow_data_da.groupby("time.dayofyear").mean(skipna=True)

            # calculate water year doy to accumulate flow for the water year vs. calendar year
            wateryear_doy = [(x+delta)%367 for x in np.arange(1,366+1)]
            for i in range(len(wateryear_doy)):
                if wateryear_doy[i] < delta:
                    wateryear_doy[i] += 1
            doy_mean = doy_mean.assign_coords(wateryear_doy=('dayofyear', wateryear_doy))
            doy_mean = doy_mean.sortby('wateryear_doy')

            # calculate the normalized cumulative sum over this climatological timeseries - we normalize so that all basins hydrographs can be plotted on a single plot
            cumsum_norm = doy_mean.cumsum() / doy_mean.cumsum().values[-1]

            # save the values to numpy array
            if basins.loc[s].regime == 'nival':
                nniv += 1
                cumsum_norm_nivalbasins[nniv,:] = cumsum_norm

            elif basins.loc[s].regime == 'glacial':
                nglac += 1
                cumsum_norm_glacialbasins[nglac,:] = cumsum_norm

        # calculate values quantiles
        niv_med = np.quantile(cumsum_norm_nivalbasins, [0,.25,.5,.75,1], axis=0)

        # plotting
        plt.plot(np.arange(366), niv_med[2,:], color='b', label='nival basins')
        plt.fill_between(np.arange(366), niv_med[1,:], niv_med[3,:], color='b', alpha=.3, lw=0)
        plt.fill_between(np.arange(366), niv_med[0,:], niv_med[4,:], color='b', alpha=.3, lw=0)

        # check that there is data for glacial basins
        if nglac > -1:

            # calculate values quantiles
            glac_med = np.quantile(cumsum_norm_glacialbasins, [0,.25,.5,.75,1], axis=0)

            # plotting
            plt.plot(np.arange(366), glac_med[2,:], color='darkturquoise', label='glacial basins')
            plt.fill_between(np.arange(366), glac_med[1,:], glac_med[3,:], color='darkturquoise', alpha=.3, lw=0)
            plt.fill_between(np.arange(366), glac_med[0,:], glac_med[4,:], color='darkturquoise', alpha=.3, lw=0)

        plt.legend()
        plt.title('')
        plt.ylabel('cdf of normalized climatological Qmean')
        plt.xticks(np.arange(0,360,30),['1st Oct', '1st Nov', '1st Dec', '1st Jan', '1st Feb', '1st Mar', '1st Apr', '1st May', '1st Jun', '1st Jul', '1st Aug', '1st Sep'], rotation=45)

    return fig;

###

def data_availability_monthly_plots_1(SWE_stations, original_SWE_data, gapfilled_SWE_data, flag):

    """Calculating and plotting the % of SWE stations available on the first day of each month of each year.

    Keyword arguments:
    ------------------
    - SWE_stations: Pandas GeoDataFrame of all SWE stations
    - original_SWE_data: xarray DataArray of the original SWE observations
    - gapfilled_SWE_data: xarray DataArray of the SWE observations after gap filling
    - flag: Flag to indicate if gap filled data was provided (1) or not (0). In the case that it is provided, a comparison plot will be made to compare data availability in the original data vs the gap filled data

    Returns:
    --------
    - Bar chart timeseries of SWE stations available on the first day of each month of each year

    """

    # Initialize plot
    fig, axs = plt.subplots(6, 2, sharex=True, sharey=True, figsize=(8,8))
    elem = -1
    column = 0

    # Loop over months
    for m in range(1,12+1):

        # controls for plotting on right subplot (i.e., month)
        elem += 1
        if elem == 6:
            column += 1
            elem = 0

        # for SWE data with gap filling
        if flag == 1:

            # extract data on the first of the month m
            data_month_gapfilled = gapfilled_SWE_data.sel(station_id=SWE_stations.station_id.values, time=( (gapfilled_SWE_data['time.month'] == m) & (gapfilled_SWE_data['time.day'] == 1) ))

            # count the % of stations with data on those dates
            data_month_gapfilled_count = data_month_gapfilled.count(dim='station_id') / len(SWE_stations) * 100

            # plot bar chart of available data
            axs[elem,column].bar(data_month_gapfilled_count['time.year'], data_month_gapfilled_count.data, color='r', alpha=.5)

        # same process as above but for original SWE data
        data_month = original_SWE_data.sel(station_id=SWE_stations.station_id.values, time=( (original_SWE_data['time.month'] == m) & (original_SWE_data['time.day'] == 1) ))
        data_month_count = data_month.count(dim='station_id') / len(SWE_stations) * 100
        axs[elem,column].bar(data_month_count['time.year'], data_month_count.data, color='b')

        # add plot labels
        if elem == 5 and column == 0:
            axs[elem,column].set_ylabel('% of SWE stations \n with data in basin')
        month_name = datetime.datetime.strptime(str(m), "%m").strftime("%b")
        axs[elem,column].set_title('1st '+month_name, fontweight='bold')

        if flag == 1:
            bluepatch = mpatches.Patch(color='b', label='original')
            redpatch = mpatches.Patch(color='r', alpha=.5, label='gap filled')
            plt.legend(handles=[bluepatch, redpatch])

    plt.tight_layout()

    return fig

###

def data_availability_monthly_plots_2(SWE_data):

    """Creating bar chart subplots of the days with SWE observations around the 1st day of each month.

    Keyword arguments:
    ------------------
    - SWE_data: Pandas DataFrame containing the SWE stations observations

    Returns:
    --------
    - Bar chart subplots of the days with SWE observations around the 1st day of each month

    """

    # Initialize plot
    fig, axs = plt.subplots(6, 2, sharex=False, sharey=True, figsize=(6,12))
    elem = -1
    column = 0

    # Add day of year (doy) to test basin SWE observations Pandas DataFrame
    SWE_data_with_doy = SWE_data.copy()
    SWE_data_with_doy['doy'] = SWE_data_with_doy.index.dayofyear

    # Remove automatic stations as they distract the analysis
    manual_stations = [s for s in SWE_data_with_doy.columns if s[-1] != 'P']
    SWE_data_with_doy_manual = SWE_data_with_doy[manual_stations]

    # Define the doys of 1st of each month
    doys_first_month = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

    # Loop over months
    for m in range(1,12+1):

        # controls for plotting on right subplot
        elem += 1
        if elem == 6:
            column += 1
            elem = 0

        # calculate the start & end of the data selection window, with caution around the start & end of the calendar year
        window_start = (doys_first_month[m-1]-15)%366
        if window_start == 0:
            window_start = 365
        window_end = (doys_first_month[m-1]+15)%366
        if window_end == 0 or window_end == 365:
            window_end = 366

        # select SWE observations within window
        if window_start > window_end:
            data_window = SWE_data_with_doy_manual[(SWE_data_with_doy_manual['doy']>=window_start) | (SWE_data_with_doy_manual['doy'] <= window_end)]
        else:
            data_window = SWE_data_with_doy_manual[(SWE_data_with_doy_manual['doy']>=window_start) & (SWE_data_with_doy_manual['doy'] <= window_end)]

        # drop dates or stations with no data at all
        data_window = data_window.dropna(axis=0, how='all')
        data_window = data_window.dropna(axis=1, how='all')

        # count total number of stations with data on each doy
        stations_cols = [c for c in data_window.columns if 'doy' not in c]
        data_stations_window = data_window[stations_cols]
        data_count_window = data_stations_window.count(axis=1)

        # create xticks to plot the data for each doy
        if window_start > window_end:
            xticks = list(np.arange(window_start,365+1))+list(np.arange(1,window_end+1))
        else:
            xticks = list(np.arange(window_start,window_end+1))
        xticks_plot = np.arange(len(xticks))

        # save the data for the right doy
        data_count_plot = [0]*len(xticks)
        for x in range(len(data_window.index)):
            doy = data_window.iloc[x]['doy']
            if doy == 366:
                doy = 365
            data_count_plot[xticks.index(doy)] += data_count_window.iloc[x]

        # plot data
        axs[elem,column].bar(xticks_plot, data_count_plot, color='b')
        axs[elem,column].set_xticks([xticks_plot[0],xticks_plot[15],xticks_plot[-1]])
        axs[elem,column].set_xticklabels([xticks[0],doys_first_month[m-1],xticks[-1]])

        # add plot labels
        if elem == 5 and column == 0:
            axs[elem,column].set_ylabel('# of SWE obs.')
            axs[elem,column].set_xlabel('DOY')

        if elem == 5 and column == 1:
            axs[elem,column].set_xlabel('DOY')

        month_name = datetime.datetime.strptime(str(m), "%m").strftime("%b")
        axs[elem,column].set_title('1st '+month_name+' +/- 15 days', fontweight='bold')

    plt.tight_layout()

    return fig

###

def deterministic_forecasting(model, test_timeseries):

    """Out-of-sample forecasting based on the test data predictor(s) and the model developed on the training data.

    Keyword arguments:
    ------------------
    - model: Regression model from statsmodels developed on the training data
    - test_timeseries: Pandas DataFrame containing the predictor & predictand data to use for testing the forecast model

    Returns:
    --------
    - flow_fc_mean: Pandas Dataframe containing the flow forecast means

    """

    # Generate out-of-sample forecasts based on the test data predictor(s)
    forecast = model.get_prediction(exog=test_timeseries)

    # Save the forecast mean to a Pandas DataFrame
    flow_fc_mean = pd.DataFrame(forecast.predicted_mean, index=test_timeseries.index, columns=["Vol_fc_mean"])

    # Set all negative forecast values to 0 as we don't want negative data
    flow_fc_mean[flow_fc_mean < 0] = 0

    return flow_fc_mean

###

def deterministic_forecast_plots(obs_timeseries, det_fc_timeseries, predictor_month, predictand_start_month, predictand_end_month, units):

    """Plot deterministic forecasts.

    Keyword arguments:
    ------------------
    - obs_timeseries: Pandas DataFrame of the observed flow accumulation data
    - det_fc_timeseries: Pandas DataFrame of the deterministic forecasts of flow accumulation
    - predictor_month: Integer of the month of predictor data to use
    - predictand_start_month: Integer of the starting month of predictand data to use
    - predictand_end_month: Integer of the end month of predictand data to use
    - units: string of the flow units to use on the plot

    Returns:
    --------
    - Timeseries and scatter plots of the flow accumulation deterministic forecasts and observations

    """

    # Select observed data that matches the forecast period
    obs_data = obs_timeseries.loc[det_fc_timeseries.index]

    # Initialize figure
    fig = plt.figure(figsize=(20, 5))
    layout = (1,3)
    ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
    scatter_ax = plt.subplot2grid(layout, (0,2))

    # Timeseries plot of deterministic forecasts and observations
    ts_ax.plot(obs_data.index, obs_data.values, color='red', label='observations', marker='o')
    ts_ax.plot(det_fc_timeseries.index, det_fc_timeseries['Vol_fc_mean'].values, color='blue', label='deterministic forecasts', marker='o')
    ts_ax.set_ylabel('Flow accumulation '+predictand_start_month+'-'+predictand_end_month+' ['+units+']')
    ts_ax.set_xticks(obs_data.index)
    ts_ax.legend()
    ts_ax.set_title('Forecasts initialized on 1st '+predictor_month)

    # Scatter plot of deterministic forecasts vs observations
    scatter_ax.scatter(obs_data.values, det_fc_timeseries['Vol_fc_mean'].values, color='k')
    min_value = min(obs_data.min(),det_fc_timeseries['Vol_fc_mean'].min())
    max_value = max(obs_data.max(),det_fc_timeseries['Vol_fc_mean'].max())
    scatter_ax.plot([min_value,max_value], [min_value,max_value], color='k', alpha=.3, ls='--')
    scatter_ax.set_xlabel('Observations'+' ['+units+']')
    scatter_ax.set_ylabel('Deterministic forecasts'+' ['+units+']')
    scatter_ax.set_title('Forecasts initialized on 1st '+predictor_month);

###

def det_metrics_calculation(Qobs, Qfc_det, flag, niterations, min_obs):

    """Calculates deterministic metrics for whole hindcast timeseries (1 value per hindcast start date & target period).

    Keyword arguments:
    ------------------
    - Qobs: xarray Dataset containing timeseries of observed flow values for various target periods of flow accumulation.
    - Qfc_det: xarray Dataset containing timeseries of deterministic flow forecasts for various target periods and forecast start dates.
    - flag: Integer to indicate whether the metrics should be calculated without (0) or with (1) bootstrapping
    - niterations: Integer > 0 of the number of bootstrapping iterations to perform
    - min_obs: Positive integer for the number of minimum number of observation-hindcast pairs required to calculate the various metrics

    Returns:
    --------
    - rsquared_da: xarray DataArray of squared correlation coefficient values for various forecast start dates & target periods
    - kge_da: xarray DataArray of KGE" values for various forecast start dates & target periods
    - perc_diff_da: xarray DataArray of percentage difference values for various forecast start dates & target periods

    """

    # Read the start dates & target periods we will be calculating metrics for
    initdates = Qfc_det.init_date.values
    targetperiods = list(Qfc_det.keys())

    # Initialize the verification metrics' Numpy arrays
    arrays = []
    for a in range(6):
        # without bootstrapping
        if flag == 0:
            arr = np.ones((len(initdates),len(targetperiods))) * np.nan
        # with bootstrapping
        elif flag == 1:
            arr = np.ones((len(initdates),len(targetperiods),niterations)) * np.nan
        arrays.append(arr)
    rsquared_array, kge_array, kge_r_array, kge_alpha_array, kge_beta_array, perc_diff_array = arrays

    # Initialize the row counter to save results in the right place
    row = -1

    # Loop over forecast starting dates
    for i in initdates:

        # increment/initialize the row/column counters to save results in the right place
        row += 1
        column = - 1

        # loop over target periods
        for t in targetperiods:

            # increment the column counter to save results in the right place
            column += 1

            # read forecasts & observations
            Qfc_det_data = Qfc_det[t].sel(init_date=i).dropna(dim='year')
            Qobs_data = Qobs[t].sel(year=Qfc_det_data.year)

            # if there are no forecasts we skip this start date-target period
            if np.isnan(Qfc_det_data.values).all() == True:
                continue

            # else, we can proceed with the verification metrics calculations
            else:

                if flag == 0:
                    # R2
                    rsquared_array[row,column] = round(corr_coeff_squared(Qobs_data.values, Qfc_det_data.values, min_obs),2)
                    # KGE" and its decomposition
                    kge_decomposition = KGE_C2021(Qobs_data.values, Qfc_det_data.values, min_obs)
                    kge_array[row,column] = round(kge_decomposition['KGE'],2)
                    kge_r_array[row,column] = round(kge_decomposition['r'],2)
                    kge_alpha_array[row,column] = round(kge_decomposition['alpha'],2)
                    kge_beta_array[row,column] = round(kge_decomposition['beta'],2)
                    # Percentage difference
                    perc_diff_array[row,column] = round(perc_difference(Qobs_data.values, Qfc_det_data.values, min_obs),2)

                elif flag == 1:

                    for b in range(niterations):

                        # random samples for bootstrapping with replacement
                        samples = np.random.choice(len(Qobs_data), len(Qobs_data))
                        Qfc_det_data_bs = Qfc_det_data.isel(year=samples)
                        Qobs_data_bs = Qobs_data.isel(year=samples)

                        # R2
                        rsquared_array[row,column,b] = round(corr_coeff_squared(Qobs_data_bs.values, Qfc_det_data_bs.values, min_obs),2)
                        # KGE" and its decomposition
                        kge_decomposition = KGE_C2021(Qobs_data_bs.values, Qfc_det_data_bs.values, min_obs)
                        kge_array[row,column,b] = round(kge_decomposition['KGE'],2)
                        kge_r_array[row,column,b] = round(kge_decomposition['r'],2)
                        kge_alpha_array[row,column,b] = round(kge_decomposition['alpha'],2)
                        kge_beta_array[row,column,b] = round(kge_decomposition['beta'],2)
                        # Percentage difference
                        perc_diff_array[row,column,b] = round(perc_difference(Qobs_data_bs.values, Qfc_det_data_bs.values, min_obs),2)

    # Save values to xarray DataArrays
    if flag == 0:
        rsquared_da = xr.DataArray(data=rsquared_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods]}, dims=['init_date','target_period'], name='R2')
        kge_da = xr.DataArray(data=kge_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods]}, dims=['init_date','target_period'], name='KGE"')
        perc_diff_da = xr.DataArray(data=perc_diff_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods]}, dims=['init_date','target_period'], name='perc_diff')

    elif flag == 1:
        rsquared_da = xr.DataArray(data=rsquared_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods],'iteration':np.arange(1,niterations+1)}, dims=['init_date','target_period','iteration'], name='R2')
        kge_da = xr.DataArray(data=kge_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods],'iteration':np.arange(1,niterations+1)}, dims=['init_date','target_period','iteration'], name='KGE"')
        kge_r_da = xr.DataArray(data=kge_r_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods],'iteration':np.arange(1,niterations+1)}, dims=['init_date','target_period','iteration'], name='KGE"_r')
        kge_alpha_da = xr.DataArray(data=kge_alpha_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods],'iteration':np.arange(1,niterations+1)}, dims=['init_date','target_period','iteration'], name='KGE"_alpha')
        kge_beta_da = xr.DataArray(data=kge_beta_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods],'iteration':np.arange(1,niterations+1)}, dims=['init_date','target_period','iteration'], name='KGE"_beta')
        perc_diff_da = xr.DataArray(data=perc_diff_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods],'iteration':np.arange(1,niterations+1)}, dims=['init_date','target_period','iteration'], name='perc_diff')

    # Information for the output xarray DataArrays
    da_dict = {'R2':rsquared_da,'KGE"':kge_da,'KGE"_r':kge_r_da,'KGE"_alpha':kge_alpha_da,'KGE"_beta':kge_beta_da,'perc_diff':perc_diff_da}
    metrics_longnames_dict = {'R2':'Squared correlation coefficient',
                              'KGE"':'Modified Kling Gupta Efficiency',
                              'KGE"_r':'Correlation',
                              'KGE"_alpha':'Variability',
                              'KGE"_beta':'Bias',
                              'perc_diff':'Percentage difference'}
    metrics_info_dict = {'R2':'Measures the linear association between deterministic hindcasts (medians) & observations. Range: -1 to 1. Perfect score: 1. Units: Unitless.',
                         'KGE"':'Combined measure of the correlation, bias & variability between deterministic hindcasts (medians) & observations. Range: -Inf to 1. Perfect score: 1. Units: Unitless.',
                         'KGE"_r':'Measures the correlation between deterministic hindcasts (medians) & observations. Perfect score: 1. Units: Unitless.',
                         'KGE"_alpha':'Measures the variability between deterministic hindcasts (medians) & observations. Perfect score: 1. Units: Unitless.',
                         'KGE"_beta':'Measures the bias between deterministic hindcasts (medians) & observations. Perfect score: 0. Units: Unitless.',
                         'perc_diff':'Measures the average forecast error between deterministic hindcasts (medians) & observations. Range: -Inf to +Inf. Perfect score: 0. Units: Unitless [%].'}

    if flag == 1:
        for keys in da_dict.keys():
            da_dict[keys].iteration.attrs['long_name'] = 'Bootstrapping iterations'
            da_dict[keys].iteration.attrs['info'] = 'We pick random samples of the datasets for bootstrapping with replacement in order to capture uncertainties in the verification metric estimates.'

    # Add information to output xarray DataArrays
    for keys in da_dict.keys():
        da_dict[keys].init_date.attrs['long_name'] = 'Hindcast initialization date'
        da_dict[keys].init_date.attrs['info'] = 'DD/MM of the predictors used to generate the hindcasts.'
        da_dict[keys].target_period.attrs['long_name'] = 'Hindcast target period'
        da_dict[keys].target_period.attrs['info'] = 'Time period for which volumes are being forecasted.'
        da_dict[keys].attrs['long_name'] = metrics_longnames_dict[keys]
        da_dict[keys].attrs['info'] = metrics_info_dict[keys]

    return rsquared_da, kge_da, kge_r_da, kge_alpha_da, kge_beta_da, perc_diff_da

###

def ensemble_dressing(det_fc, SD, ens_size):

    """Generate ensembles around deterministic forecasts. They are generated by drawing random samples from a normal (Gaussian) distribution.

    Keyword arguments:
    ------------------
    - det_fc: Pandas DataFrame of the deterministic forecasts
    - SD: Positive value of the standard deviation of errors between the forecast means and observed values during training
    - ens_size: Integer > 0 of the number of ensemble members to generate

    Returns:
    --------
    - ens_fc: Pandas DataFrame of the ensemble forecasts produced

    """

    # Initialize empty Numpy array to store the ensemble member values
    ensembles = np.zeros((len(det_fc.index), ens_size))

    # Loop over forecasts
    for x in range(len(det_fc.index)):

        # generate randomly spaced ensemble members
        ensembles[x,:] = np.random.normal(loc=det_fc['Vol_fc_mean'].values[x], scale=SD, size=ens_size)

    # Save ensemble forecasts to Pandas DataFrame
    ens_fc = pd.DataFrame(data=ensembles, index=det_fc.index, columns=np.arange(1,ens_size+1))

    # Set all negative forecast values to 0
    ens_fc[ens_fc < 0] = 0

    return ens_fc

###

def ensemble_forecasting(predictor_data, predictand_data, PC_ids, ens_size, min_overlap_years, method_traintest, nyears_leaveout):

    """Generate ensemble forecasts of flow accumulations (predictand) from SWE PC(s) (predictors).

    Keyword arguments:
    ------------------
    - predictor_data: Pandas DataFrame of the predictor data
    - predictand_data: Pandas DataFrame of the predictand data
    - PC_ids: String (if only 1) or list (if > 1) of the PC(s) to use as predictor data
    - ens_size: Integer > 0 of the number of ensemble members to generate
    - min_overlap_years: Positive integer indicating the minimum number of years required of PC-volume to be able to generate a forecast
    - method_traintest: String to define the method used to split the data into training and testing samples
    - nyears_leaveout: Integer above zero for the number of years to leave out at a time

    Returns:
    --------
    - fc_ens_df: Pandas dataframe containing all generated ensemble hindcasts

    """

    # Clean predictor and predictand datasets and find the number of overlapping years with data
    cleaned_predictor_data = predictor_data.dropna(axis=1,thresh=min_overlap_years).dropna(axis=0,how='any')
    cleaned_predictand_data = predictand_data.dropna()
    if (cleaned_predictor_data.empty == False) and (cleaned_predictand_data.empty == False):
        cleaned_predictor_data_years = cleaned_predictor_data.index.year.values
        cleaned_predictand_data_years = cleaned_predictand_data.index.values
        overlapping_years = list(set(cleaned_predictor_data_years) & set(cleaned_predictand_data_years))
        overlapping_years.sort()
    else:
        overlapping_years = []

    # If there is enough data we can carry on with the forecasting
    if (len(overlapping_years)-nyears_leaveout) >= min_overlap_years:

        overlapping_predictor_data = cleaned_predictor_data[cleaned_predictor_data.index.year.isin(overlapping_years)]
        overlapping_predictand_data = predictand_data.loc[overlapping_years]

        # Run PCA
        PCs, loadings = principal_component_analysis(overlapping_predictor_data, flag=0)

        combined_df = PCs.reset_index(drop=True)
        combined_df['year'] = overlapping_years
        combined_df = combined_df.set_index('year')
        combined_df['Vol'] = overlapping_predictand_data

        # remaining forecasting steps will depend on the approach selected to split the data into train-test samples

        # Leave X years out approach
        if method_traintest == 'leave_out':

            # Split the timeseries into training and validation timeseries for forecasting
            train_data_dict, test_data_dict = leave_out(combined_df, nyears_leaveout)

            # Loop over the samples
            for s in list(train_data_dict.keys()):

                # Select train and test data
                train_data = train_data_dict[s]
                test_data = test_data_dict[s]

                # fit the model on the training data
                OLS_model = OLS_model_fitting(PC_ids, train_data)

                # perform out-of-sample deterministic forecasting for the testing period
                fc_det = deterministic_forecasting(OLS_model, test_data)

                # calculate errors standard deviation for the training period
                fc_det_train = deterministic_forecasting(OLS_model, train_data)
                rmse = mean_squared_error(train_data['Vol'].values, fc_det_train['Vol_fc_mean'].values, squared=False)

                # generate ensembles
                fc_ens = ensemble_dressing(fc_det, rmse, ens_size=ens_size)

                # append all ensembles generated for each moving window
                if s == 0:
                    fc_ens_df = fc_ens
                else:
                    fc_ens_df = pd.concat([fc_ens_df,fc_ens])

        return fc_ens_df

    # Otherwise return nothing
    else:
        return None

###

def ensemble_forecast_plots(obs_timeseries, ens_fc_timeseries, predictor_month, predictand_target_period, units):

    """Plot ensemble forecasts timeseries as boxplots.

    Keyword arguments:
    ------------------
    - obs_timeseries: Pandas DataFrame of the observed flow accumulation data
    - ens_fc_timeseries: Pandas DataFrame of the ensemble forecasts of flow accumulation
    - predictor_month: String of the month of predictor data to use
    - predictand_target_period: String of the predictand target period to use
    - units: string of the flow units to use on the plot

    Returns:
    --------
    - Timeseries plot of the flow accumulation ensemble forecasts and observations

    """

    # Select observed data that matches the forecast period
    ens_fc_timeseries = ens_fc_timeseries.dropna(dim='year')
    obs_data = obs_timeseries.loc[ens_fc_timeseries.year]

    # Initialize figure
    fig = plt.figure(figsize=(20, 5))
    ts_ax = plt.subplot()

    # Timeseries plot of ensemble forecasts and observations
    reddots, = ts_ax.plot(np.arange(1, len(obs_data)+1), obs_data.values, color='red', label='observations', marker='o')
    bp = plt.boxplot(np.transpose(ens_fc_timeseries.values), patch_artist=True, zorder=1, whis=[0, 100], showfliers=False)
    plt.setp(bp['boxes'], color='b', alpha=.5)
    plt.setp(bp['whiskers'], color='b')
    plt.setp(bp['medians'], color='k')
    bluepatch = mpatches.Patch(color='b', alpha=.5, label='ensemble forecasts')
    ts_ax.set_ylabel('Flow accumulation '+predictand_target_period+' ['+units+']')
    ts_ax.set_xticks(np.arange(1, len(obs_data)+1))
    ts_ax.set_xticklabels(obs_data.index.values)
    plt.legend(handles=[reddots,bluepatch])
    ts_ax.set_title('Forecasts initialized on 1st '+predictor_month)

    return fig

###

def extract_monthly_data(stations_data, month, flag):

    """For the PCA & forecasting, we need a full dataset (no missing data) for specific dates.
    For our use, we extract data (with no missing values) for the first day of a given month.
    We find the optimal number of stations and years of data we keep.

    Keyword arguments:
    ------------------
    - stations_data: Pandas DataFrame containing the (gapfilled) SWE stations observations
    - month: Integer between 1 and 12 to specify the month for which we want to extract data (1st day of the month extracted)
    - flag: Integer to plot the evolution of the selection criteria (1) or not (0)

    Returns:
    --------
    - month_stations_final: Pandas DataFrame containing the SWE stations observations to keep
    - optional plot of the evolution of the # of stations & years we can keep

    """

    # Select all stations data for 1st of the month
    month_stations_data = stations_data[(stations_data.index.month == month) & (stations_data.index.day == 1)]

    # Create empty lists to store the # of stations and the # of years with data
    no_stations = []
    no_dates = []

    # Increase step by step the minimum # of (non-missing) values we need to keep a station
    for t in range(len(month_stations_data.index)+1):

        # drop stations that do not meet this minimum threshold of values
        month_stations_test = month_stations_data.dropna(axis=1, thresh=t)

        # drop years with any missing values across remaining stations
        month_stations_test = month_stations_test.dropna(axis=0, how='any')

        # calculate and save the # of stations and years with data remaining
        if len(month_stations_test.count(axis=1)) == 0 or len(month_stations_test.count(axis=0)) == 0:
            no_stations.append(0)
            no_dates.append(0)
        else:
            no_stations.append(month_stations_test.count(axis=1)[0])
            no_dates.append(month_stations_test.count(axis=0)[0])

    # multiply the # of stations remaining by the # of years remaining to find the optimal combination
    products = []
    for num1, num2 in zip(no_stations, no_dates):
        products.append(num1 * num2)

    # the optimal threshold is the threshold corresponding to the 1st occurence of max. stations x years available
    thresh_optimal = products.index(max(products))

    # select data for the optimal # stations & years combination
    month_stations_final = month_stations_data.dropna(axis=1, thresh=thresh_optimal)
    month_stations_final = month_stations_final.dropna(axis=0, how='any')

    # plot the evolution of the # of stations & years of data
    if flag == 1:
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        ax1.plot(np.arange(len(no_stations)), no_stations, marker='o', color='b')
        ax2.plot(np.arange(len(no_dates)), no_dates, marker='o', color='r')
        ax1.plot([thresh_optimal]*2, [0,max(no_stations)], color='grey', ls='--', label='optimal')
        ax2.plot([thresh_optimal]*2, [0,max(no_dates)], color='grey', ls='--', label='optimal')
        plt.xlabel('min. # of values per station requires')
        ax1.set_ylabel('# stations left')
        ax2.set_ylabel('# years left')
        plt.legend();

    return month_stations_final

###

def extract_stations_in_basin(stations, basins, basin_id, buffer_km=0):

    """Extracts stations within a specified basin (with or without a buffer) and returns the extracted stations.

    Keyword arguments:
    ------------------
    - stations: Pandas GeoDataFrame of all stations available to subset from
    - basins: Pandas GeoDataFrame of all basin shapefiles available to subset from
    - basin_id: String of basin station ID
    - buffer_km: Positive value (in km) of buffer to add around basin shapefile (default=0; i.e., no buffer)

    Returns:
    --------
    - stations_in_basin: Pandas GeoDataFrame of all stations within the specified basin
    - basin_buffer: Zero if the default buffer is selected, otherwise buffer geometry for plotting

    """

    # Extract SWE stations within basin only (i.e., no buffer)
    if buffer_km == 0:
        basin_buffer = 0
        mask = stations.within(basins.loc[basins['Station_ID'] == basin_id].iloc[0].loc["geometry"])

    # Extract SWE stations within specified buffer of basin
    elif buffer_km > 0:

        # convert basin & stations geometry to a different CRS to be able to add a buffer in meters
        basin_crs_conversion = basins.loc[basins['Station_ID'] == basin_id].to_crs(epsg=3763)
        stations_crs_conversion = stations.to_crs(epsg=3763)

        # add a buffer in meters around the basin
        buffer_m = buffer_km * 1000
        basin_buffer = basin_crs_conversion.buffer(buffer_m)
        mask = stations_crs_conversion.within(basin_buffer.iloc[0])

        # convert the buffer back to the original CRS for plotting
        basin_buffer = basin_buffer.to_crs(epsg=4326)

    stations_in_basin = stations.loc[mask].assign(basin=basin_id)

    return stations_in_basin, basin_buffer

###

def hydrographs(streamflow_obs, month_start_water_year, day_start_water_year):

    """Plots normalized climatological streamflows for the provided basins, differenciating between nival and glacial basins.

    Keyword arguments:
    ------------------
    - streamflow_obs: xarray Dataset of streamflow observations
    - month_start_water_year: Integer of the water year starting month
    - day_start_water_year: Integer of the water year starting day of the month

    Returns:
    --------
    - A plot of all basins' climatological hydrographs.

    """
    # Set up figure
    fig = plt.figure(figsize=(5,4))

    # We expect to see RuntimeWarnings in this block due to missing values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # calculate the number of days between the water year and calendar start
        start_water_year = date(2020,month_start_water_year,day_start_water_year)
        start_calendar_year = date(2021,1,1)
        delta = (start_calendar_year - start_water_year).days

        # extract all stations for plotting
        stations_to_plot = [x for x in streamflow_obs.Station_ID.values]

        # set empty array to save values for plotting if there are more than 1 station to plot
        if len(stations_to_plot) > 1:
            norm_basins = np.ones((len(stations_to_plot),366))

        # set counter
        n = -1

        # loop over the stations
        for s in stations_to_plot:

            n += 1

            # read streamflow observations for basin as xarray DataArray
            if len(stations_to_plot) == 1:
                streamflow_data_da = streamflow_obs.Flow
            else:
                streamflow_data_da = streamflow_obs.sel(Station_ID=s).Flow

            # calculate the climatological mean flow for each DOY
            doy_mean = streamflow_data_da.groupby("time.dayofyear").mean(skipna=True)

            # calculate water year doy to accumulate flow for the water year vs. calendar year
            wateryear_doy = [(x+delta)%367 for x in np.arange(1,366+1)]
            for i in range(len(wateryear_doy)):
                if wateryear_doy[i] < delta:
                    wateryear_doy[i] += 1
            doy_mean = doy_mean.assign_coords(wateryear_doy=('dayofyear', wateryear_doy))
            doy_mean = doy_mean.sortby('wateryear_doy')

            # calculate the normalized sum over this climatological timeseries - we normalize so that all basins hydrographs can be plotted on a single plot
            norm = doy_mean / doy_mean.cumsum().values[-1]

            if len(stations_to_plot) > 1:
                # save the values to numpy array
                norm_basins[n,:] = norm

        # if there is only 1 station to plot
        if len(stations_to_plot) == 1:
            # plotting
            plt.plot(np.arange(366), norm, color='b')

        # if there are more than 1 station to plot
        elif len(stations_to_plot) > 1:
            # calculate values quantiles
            # Note: we start at .1 and end at .9 to remove noise with extremes
            quant = np.quantile(norm_basins, [.1,.25,.5,.75,.9], axis=0)
            # plotting
            plt.plot(np.arange(366), quant[2,:], color='b', label='median')
            plt.fill_between(np.arange(366), quant[1,:], quant[3,:], color='b', alpha=.5, lw=0, label='25th to 75th perc. range')
            plt.fill_between(np.arange(366), quant[0,:], quant[1,:], color='b', alpha=.3, lw=0, label='10th to 90th perc. range')
            plt.fill_between(np.arange(366), quant[3,:], quant[4,:], color='b', alpha=.3)
            plt.legend()

        # add plot elements
        plt.title('')
        plt.ylabel('normalized climatological Qmean')
        plt.xticks(np.arange(0,360,30),['1st Oct', '1st Nov', '1st Dec', '1st Jan', '1st Feb', '1st Mar', '1st Apr', '1st May', '1st Jun', '1st Jul', '1st Aug', '1st Sep'], rotation=45)
        plt.tight_layout()

    return fig;

###

def KGE_C2021(obs, pre, min_obs_KGE):

    """Calculates the modified Kling-Gupta Efficiency (KGE") and its 3 components.
    The KGE measures the correlation, bias and variability of the simulated values against the observed values.
    KGE" was proposed by Clark et al. (2021) to solve issues arising with 0 values in the KGE or KGE'.
    For more info, see Clark et al. (2021): https://doi.org/10.1029/2020WR029001
    KGE" range: -Inf to 1. Perfect score: 1. Units: Unitless.
    Correlation (r): Perfect score is 1.
    Bias (beta): Perfect score is 0.
    Variability (alpha):  Perfect score is 1.

    Keyword arguments:
    ------------------
    - obs: Numpy Array of observations to evaluate
    - pre: Numpy Array of predictions/simulations to evaluate
    - min_obs_KGE: Positive integer for the minimum number of stations required to calculate a station's cdf

    Returns:
    --------
    - KGEgroup: Dictionary containing the final KGE'' value as well as all elements of the KGE''

    """

    ind_nan = np.isnan(obs) | np.isnan(pre)
    obs = obs[~ind_nan]
    pre = pre[~ind_nan]

    if len(obs) >= min_obs_KGE:

        pre_mean = np.mean(pre, axis=0, dtype=np.float64)
        obs_mean = np.mean(obs, axis=0, dtype=np.float64)
        pre_std = np.std(pre, axis=0)
        obs_std = np.std(obs, dtype=np.float64)

        # Check to see if all forecast values are the same. If they are r cannot be calculated and is set to 0
        # For more info: https://doi.org/10.5194/hess-23-4323-2019 (Section 2)
        if pre_std == 0:

            r = 0

        else:

            r = np.sum((pre - pre_mean) * (obs - obs_mean), axis=0, dtype=np.float64) / \
                np.sqrt(np.sum((pre - pre_mean) ** 2, axis=0, dtype=np.float64) *
                        np.sum((obs - obs_mean) ** 2, dtype=np.float64))

        alpha = pre_std / obs_std

        beta = (pre_mean - obs_mean) / obs_std

        KGE = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta) ** 2)

        KGEgroup = {'KGE': KGE, 'r': r, 'alpha': alpha, 'beta': beta}

    else:

        KGEgroup = {'KGE': np.nan, 'r': np.nan, 'alpha': np.nan, 'beta': np.nan}

    return KGEgroup

###

def leave_out(original_timeseries, nyears_leaveout):

    """Splits predictor & predictand timeseries data using leave years out method (training & testing the forecast model).
    E.g., if nyears_leaveout = 1, we leave one year out each time for which we want to test (i.e., validate) the model. All other years will be used for training the model.
    If nyears_leaveout = 3, we leave 3 successive years out at a time.

    Keyword arguments:
    ------------------
    - original_timeseries: Pandas DataFrame of the combined predictor & predictand timeseries
    - nyears_leaveout: Integer above zero for the number of years to leave out at a time

    Returns:
    --------
    - train_timeseries: Pandas DataFrame containing the 1st half of the timeseries used for training the forecast model
    - test_timeseries: Pandas DataFrame containing the 2nd half of the timeseries used for testing the forecast model

    """

    n_examples = len(original_timeseries)
    test_dict = {}
    train_dict = {}
    k=0
    remainder = n_examples % nyears_leaveout

    while(k * nyears_leaveout + nyears_leaveout <= n_examples):
        test_samples = list(range(k * nyears_leaveout,k * nyears_leaveout + nyears_leaveout))
        test_dict[k] = original_timeseries.iloc[test_samples]
        train_samples = [x for x in list(range(n_examples)) if x not in test_samples]
        train_dict[k] = original_timeseries.iloc[train_samples]
        k+=1

    if remainder != 0:
        test_samples = list(range(n_examples-remainder,n_examples))
        test_dict[k] = original_timeseries.iloc[test_samples]
        train_samples = [x for x in list(range(n_examples)) if x not in test_samples]
        train_dict[k] = original_timeseries.iloc[train_samples]

    return train_dict, test_dict

###

def maps_loadings(dem_dir, basin_id, basin, SWE_stations, loadings, PC):

    """Creating maps of loadings between basin SWE stations and a given PC for each 1st of the month to see correlations in time & space.

    Keyword arguments:
    ------------------
    - dem_dir: String of the path to DEMs
    - basin_id: String of the basin id to plot
    - basin: Pandas GeoDataFrame of basin to plot
    - SWE_stations: Pandas GeoDataFrame of SWE stations to plot
    - loadings: Dictionary of the PCA loadings (correlation between PCs & stations data)
    - PC: String of the PC to create the maps for (e.g., 'PC1')

    Returns:
    --------
    - Maps of loadings between basin SWE stations and a given PC for each 1st of the month

    """

    # Initialize figure
    fig, ax = plt.subplots(4,3, figsize=[10,30])
    plot_col = -1
    row = 0

    # Load DEM
    src = rasterio.open(dem_dir + basin_id[:2] + ".tif")

    # Loop over months
    for m in range(1,12+1):

        # controls for plotting on right subplot
        plot_col += 1
        if plot_col == 3:
            row += 1
            plot_col = 0

        # month name to string
        month_name = datetime.datetime.strptime(str(m), "%m").strftime("%b")

        # add basin contour & elevation shading to map
        basin.plot(ax=ax[row,plot_col], edgecolor='k', facecolor='none', lw=.5)
        rasterio.plot.show((src, 1), cmap='Greys', vmin=0, ax=ax[row,plot_col], alpha=.7)

        # remove frame ticks
        ax[row,plot_col].set_xticks([])
        ax[row,plot_col].set_yticks([])

        # calculate map bounds
        minx, miny, maxx, maxy = basin.geometry.total_bounds
        ax[row,plot_col].set_xlim(minx - .1, maxx + .1)
        ax[row,plot_col].set_ylim(miny - .1, maxy + .1)

        # add labels
        ax[row,plot_col].set_title('1st '+month_name)

        # check if there is data for this month:
        if m in loadings:

            # extract geospatial information for stations to plot
            SWE_stations_extract = SWE_stations[SWE_stations.station_id.isin(loadings[m].columns)]
            SWE_stations_extract = SWE_stations_extract.set_index('station_id')

            # extract & plot data for PC if available
            if PC in loadings[m].index:
                data_for_map = SWE_stations_extract.merge(loadings[m].loc[PC], left_index=True, right_index=True, how='outer')
                sc = ax[row,plot_col].scatter(data_for_map.lon.values, data_for_map.lat.values, c=data_for_map[PC].values, cmap='rocket_r', vmin=0, vmax=1)

    # Adjust subplots & add colorbar
    plt.tight_layout()
    fig.subplots_adjust(hspace=-.85)
    cbar_ax = fig.add_axes([1.01,.31,.01,.072])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('R$^2$')

    return fig

###

def metrics_bootstrap_plots(metric_values, min_value, max_value, flag_skill, flag_events):

    """Plots metrics median values with confidence intervals from bootstrapping.

    Keyword arguments:
    ------------------
    - metric_values: xarray DataArray containing verification metric values for various target periods and forecast start dates.
    - min_value: Minimum value of that verification metric
    - max_value: Maximum value of that verification metric
    - flag_skill: Integer to indicate whether to plot a 0 value threshold line (1) or not (0)
    - flag_events: Integer to indicate whether to plot 1 score (0) or 2 scores/events to compare (1)

    Returns:
    --------
    - Sub-plots of the evolution of the verification metric values per start date for each target period.

    """

    # Initialize figure
    nrows = int(len(metric_values.target_period)/3+len(metric_values.target_period)%3)
    fig, ax = plt.subplots(3,nrows, figsize=[10,10])

    # Initialize counter to create subplots in the right place
    elem = -1

    # Loop over forecast target periods
    for t in metric_values.target_period.values:

        # increment counter
        elem += 1

        # plot metric median values and bootstrapping confidence intervals
        if flag_events == 1:
            ax[int(np.floor(elem/3)),elem%3].plot(np.arange(1,len(metric_values.init_month)+1), metric_values.sel(target_period=t).isel(event=0).median(dim='iteration'), color='orange', marker='o', label='median ($\leq$'+str(int(metric_values.event[0]*100))+'th perc.)')
            ax[int(np.floor(elem/3)),elem%3].plot(np.arange(1,len(metric_values.init_month)+1), metric_values.sel(target_period=t).isel(event=1).median(dim='iteration'), color='b', marker='o', label='median ($\geq$'+str(int(metric_values.event[1]*100))+'th perc.)')
            ax[int(np.floor(elem/3)),elem%3].fill_between(np.arange(1,len(metric_values.init_month)+1), metric_values.sel(target_period=t).isel(event=0).min(dim='iteration'), metric_values.sel(target_period=t).isel(event=0).max(dim='iteration'), color='orange', alpha=.1, label='conf. int. ($\leq$'+str(int(metric_values.event[0]*100))+'th perc.)')
            ax[int(np.floor(elem/3)),elem%3].fill_between(np.arange(1,len(metric_values.init_month)+1), metric_values.sel(target_period=t).isel(event=1).min(dim='iteration'), metric_values.sel(target_period=t).isel(event=1).max(dim='iteration'), color='b', alpha=.1, label='conf. int. ($\geq$'+str(int(metric_values.event[1]*100))+'th perc.)')
        else:
            ax[int(np.floor(elem/3)),elem%3].plot(np.arange(1,len(metric_values.init_month)+1), metric_values.sel(target_period=t).median(dim='iteration'), color='b', marker='o', label='median')
            ax[int(np.floor(elem/3)),elem%3].fill_between(np.arange(1,len(metric_values.init_month)+1), metric_values.sel(target_period=t).min(dim='iteration'), metric_values.sel(target_period=t).max(dim='iteration'), color='grey', alpha=.1, label='conf. int.')

        # plot threshold line
        if flag_skill == 1:
            ax[int(np.floor(elem/3)),elem%3].plot(np.arange(1-.5,len(metric_values.init_month)+1+.5), [0]*(len(metric_values.init_month)+1), color='grey', ls='--', lw=.5)

        # Add labels & legend
        ax[int(np.floor(elem/3)),elem%3].set_xticks(metric_values.init_month)
        ax[int(np.floor(elem/3)),elem%3].set_xlim(1-.5,len(metric_values.init_month)+.5)
        ax[int(np.floor(elem/3)),elem%3].set_ylim(min_value, max_value)
        ax[int(np.floor(elem/3)),elem%3].set_title(t)
        if int(np.floor(elem/3)) == nrows-1:
            ax[int(np.floor(elem/3)),elem%3].set_xticklabels([datetime.datetime.strptime(str(x), "%m").strftime("%b") for x in metric_values.init_month.values], rotation=90)
            ax[int(np.floor(elem/3)),elem%3].set_xlabel('Hindcast start dates (1st)')
            if elem%3 == 0:
                ax[int(np.floor(elem/3)),elem%3].legend()
                if metric_values.name == 'R2':
                    ax[int(np.floor(elem/3)),elem%3].set_ylabel('R$^2$')
                elif metric_values.name == 'ROC_AUC':
                    ax[int(np.floor(elem/3)),elem%3].set_ylabel('ROC AUC')
                else:
                    ax[int(np.floor(elem/3)),elem%3].set_ylabel(metric_values.name)
        else:
            ax[int(np.floor(elem/3)),elem%3].set_xticklabels([])
        if elem%3 != 0:
            ax[int(np.floor(elem/3)),elem%3].set_yticklabels([]);

###

def OLS_model_fitting(PC_ids, train_timeseries):

    """Fits the OLS model using the specified predictor(s) and training data.

    Keyword arguments:
    ------------------
    - PC_ids: String (if only 1) or list (if more than 1) of the PC(s) to use as predictor data
    - train_timeseries: Pandas DataFrame containing the predictor & predictand data to use for training the forecast model

    Returns:
    --------
    - model_fit: OLS model

    """

    # Initialize formula used in the forecast model
    formula = 'Vol ~ '

    # Build formula by adding predictor(s)
    # If more than one predictor
    if type(PC_ids) == list:

        # make sure PC we're calling exists in the dataset
        PCs_to_use = tuple(set.intersection(set(train_timeseries.columns.values), set(tuple(PC_ids))))

        # add PC to formula
        formula += PCs_to_use[0]
        for pc in PCs_to_use[1::]:
            formula += ' + ' + pc

    # If only one predictor
    else:
        formula += PC_ids

    # Fit OLS model on the training dataset
    OLS_model = smf.ols(formula=formula, data=train_timeseries)
    model_fit = OLS_model.fit()

    return model_fit

###

def perc_difference(Qobs, Qfc_det, min_obs):

    """Calculates the percentage difference between deterministic forecasts & observations.
    Answers the question: What is the average forecast error?
    Range: -Inf to +Inf. Perfect score: 0. Units: Unitless [%].
    Characteristics: Says something about bias, a simple & familiar emasure. Note that it is possible to get a perfect score for a bad forecast if there are compensating errors.

    Keyword arguments:
    ------------------
    - Qobs: Numpy Array containing a timeseries of observed flow values
    - Qfc_det: Numpy Array containing a timeseries of deterministic flow forecasts (e.g., medians or means of the ensemble forecasts)
    - min_obs: Positive integer for the number of minimum number of observation-hindcast pairs required to calculate the various metrics

    Returns:
    --------
    - perc_diff: Float of the percent difference between the deterministic forecasts & observations.

    """

    ind_nan = np.isnan(Qobs) | np.isnan(Qfc_det)
    obs = Qobs[~ind_nan]
    pre = Qfc_det[~ind_nan]

    if len(obs) >= min_obs:

        perc_diff = np.nanmean(((pre - obs) / obs) * 100)

    else:

        perc_diff = np.nan

    return perc_diff

###

def plots_artificial_gap_evaluation(evaluation_scores):

    """Plotting evaluation results for the artificial gap filling.

    Keyword arguments:
    ------------------
    - evaluation_metrics: Dictionary containing the artificial gap filling evaluation results for several metrics for each month's first day, station & iteration

    Returns:
    --------
    - plots of the various evaluation metrics for all stations, iterations & each month's first day

    """

    # Initialize figure
    ncols = 3
    fig, axs = plt.subplots(2, ncols, sharex=True, sharey=False, figsize=(9,5))
    elem = -1
    row = 0

    # Define metrics used & their optimal values
    metrics = list(evaluation_scores.keys())
    metrics_optimal_values = {'RMSE':0, "KGE''":1, "KGE''_corr":1, "KGE''_bias":0, "KGE''_var":1}
    units = {'RMSE':'mm', "KGE''":'-', "KGE''_corr":'-', "KGE''_bias":'-', "KGE''_var":'-'}

    # Loop over metrics
    for m in metrics:

        # controls for plotting on right subplot
        elem += 1
        if elem == ncols:
            row += 1
            elem = 0

        # loop over iterations
        for i in range(evaluation_scores[m].shape[2]):

            # Plot boxplot for each month
            for mo in range(1,12+1):
                nonan = evaluation_scores[m][mo-1,:,i][~np.isnan(evaluation_scores[m][mo-1,:,i])]
                bp = axs[row,elem].boxplot(nonan, positions=[mo], patch_artist=True, showfliers=False, widths=.7)
                plt.setp(bp['boxes'], color='b', alpha=.5)
                plt.setp(bp['whiskers'], color='b')
                plt.setp(bp['medians'], color='k')

            # # loop over stations
            # for s in range(evaluation_scores[m].shape[1]):
            #     # plot evaluation results
            #     axs[row,elem].scatter(np.arange(1,12+1), evaluation_scores[m][:,s,i], color='b', alpha=.3)

        # Add elements to the plot
        axs[row,elem].plot(np.arange(0,13+1), [metrics_optimal_values[m]]*14, color='grey', ls='--', label='best values')
        axs[row,elem].set_xlim([0, 13])
        axs[row,elem].set_xticks(np.arange(1,12+1))
        axs[row,elem].set_ylabel(m+' ['+units[m]+']', fontweight='bold')
        axs[row,elem].tick_params(axis='y', labelsize=8)

        if row == 1:
            axs[row,elem].set_xticklabels(np.arange(1,12+1), fontsize=8)

    axs[1,0].legend(fontsize=8)
    axs[1,0].set_xlabel('months (1st of)', fontweight='bold')
    fig.delaxes(axs[1][2])
    plt.tight_layout()

    return fig

###

def polar_plot(theta_rad, regularity, flag, nival_start_doy, nival_end_doy, nival_regularity_threshold):

    """Plots circular statistics on a polar plot for a single or multiple basins.

    Keyword arguments:
    ------------------
    - theta_rad: Numpy array of the angular values (in radians) that correspond to the doy
    - regularity: Dimensionless value indicating the spread of the data (ranges from 0: high spread to 1: low spread)
    - flag: Flag to indicate if a single (0) or multiple (1) basins should be plotted
    - nival_start_doy: Integer day of year (doy) of the start of the nival period (default=pre-defined at the top of the Notebook)
    - nival_end_doy: Integer day of year (doy) of the end of the nival period (default=pre-defined at the top of the Notebook)
    - nival_regularity_threshold: Float of the minimum regularity threshold allowed for basins to be categorized as being nival (default=pre-defined at the top of the Notebook)

    Returns:
    --------
    - A polar plot of circular statistics.

    """

    # Initialize the figure
    fig = plt.figure(figsize=(7,5))
    ax = plt.subplot(polar=True)
    lines, labels = plt.thetagrids(np.arange(0,360,30),['1st Jan', '1st Feb', '1st Mar', '1st Apr', '1st May', '1st Jun', '1st Jul', '1st Aug', '1st Sep', '1st Oct', '1st Nov', '1st Dec'])
    ax.set_theta_zero_location('S')
    ax.set_theta_direction(-1)
    ax.tick_params(pad=10)

    # Plot circular statistics for a single basin - i.e., individual events are shown
    if flag == 0:
        plt.plot(circmean(theta_rad), regularity, 'bo', label='circular statistics')
        plt.plot(theta_rad, [1.1]*len(theta_rad), 'ks', ms=5, alpha=.5, label='events');

    # Plot circular statistics for multiple basins - i.e., individual events are not shown
    elif flag == 1:
        plt.plot(theta_rad, regularity, 'bo')

    # Add information on polar plot about rainfall-driven & nival regimes
    nival_start = nival_start_doy * ((2*math.pi)/365)
    nival_end = nival_end_doy * ((2*math.pi)/365)
    plt.fill_between(np.linspace(nival_start, nival_end, 100), nival_regularity_threshold, 1, color='b', alpha=.2, label='nival')
    raindriven_start_1 = 274 * ((2*math.pi)/365)
    raindriven_end_1 = 365 * ((2*math.pi)/365)
    raindriven_start_2 = 0 * ((2*math.pi)/365)
    raindriven_end_2 = 30 * ((2*math.pi)/365)
    plt.fill_between(np.linspace(raindriven_start_1, raindriven_end_1, 100), 0.45, 1, color='g', alpha=.2, label='rainfall-driven')
    plt.fill_between(np.linspace(raindriven_start_2, raindriven_end_2, 100), 0.45, 1, color='g', alpha=.2)

    plt.legend(loc=(1,1))
    plt.tight_layout()

    return fig

###

def predictor_predictand_corr_plot(predictor_data, predictand_data, PC_id, start_months, end_month, min_obs_corr):

    """Calculates and plots correlations between a SWE PC (predictor) & flow volumes (predictand) for different lead times and for different volume accumulation periods.

    Keyword arguments:
    ------------------
    - predictor_data: Pandas DataFrame of the predictor (SWE PC) data
    - predictand_data: Pandas DataFrame of the predictand (flow columes) data
    - PC_id: String of the PC to use for the predictor data
    - start_months: List of integers of the starting months of volume accumulation periods (predictand)
    - end_month: Integer of the end month of volume accumulation periods (predictand)
    - min_obs_corr: Positive integer defining the minimum number of observations required to calculate the correlation between predictand-predictor

    Returns:
    --------
    - correlations: Pandas DataFrame of the correlations between predictors and predictands
    - A matrix plot of the correlations between predictors and predictands

    """

    # Initialize the figure
    fig = plt.figure()

    # Initialize empty Numpy array to store correlations between predictors & predictands
    corr_array = np.ones((len(start_months),len(start_months))) * np.nan

    # Initialize predictand and predictor indices for saving the correlations & plotting
    predictand_idx = []
    predictor_idx = []

    # Set the name of the end month of the predictand's accumulation periods - e.g., 'Sep' for month 9
    predictand_end_month_name = datetime.datetime.strptime(str(end_month), "%m").strftime("%b")

    # Initialize counter for saving & plotting data
    elem_row = -1

    # Loop over predictor dates
    for predictor_month in start_months:

        # update counters for saving & plotting data
        elem_row += 1
        elem_col = elem_row - 1

        # save the predictor indices to empty list
        predictor_month_name = datetime.datetime.strptime(str(predictor_month), "%m").strftime("%b")
        predictor_idx.append(predictor_month_name)

        # loop over predictand accumulation periods start dates
        for predictand_start_month in range(predictor_month, end_month+1):

            # update counters for saving & plotting data
            elem_col += 1

            # set the name of the starting month of the predictand's accumulation periods - e.g., 'Sep' for month 9
            predictand_start_month_name = datetime.datetime.strptime(str(predictand_start_month), "%m").strftime("%b")

            # save the predictand label to empty list
            if elem_row == 0:
                predictand_idx.append(predictand_start_month_name+'-'+predictand_end_month_name)

            # get predictor & predictand series for overlapping time period
            SWE_data = predictor_data[(predictor_data.index.month == predictor_month) & (predictor_data.index.day == 1)][PC_id]
            SWE_data = SWE_data.dropna()
            new_index = SWE_data.index.year
            SWE_data = pd.Series(SWE_data.values, name='SWE', index=pd.Index(new_index, name='year'))
            flow_data = predictand_data['Vol_'+predictand_start_month_name+'-'+predictand_end_month_name].rename('Qobs')

            # calculate correlations between predictor & predictand and save results to Numpy array
            corr_value = round(SWE_data.corr(flow_data, method='pearson', min_periods=min_obs_corr)**2,2)
            corr_array[elem_row, elem_col] = corr_value

    # Save correlations Numpy array as Pandas DataFrame
    correlations = pd.DataFrame(data=corr_array, index=predictor_idx, columns=predictand_idx)

    # Plot matrix of correlations
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(correlations, annot=True, cmap=cmap, cbar_kws={'label': 'R$^2$'}, vmin=0, vmax=1)
    ax.set(xlabel="Flow accumulation periods", ylabel = "SWE dates (1st)")

    return correlations, fig

###

def principal_component_analysis(stations_data, flag):

    """Transforming stations observations into principal components.

    Keyword arguments:
    ------------------
    - stations_data: Pandas DataFrame containing the (gapfilled) SWE stations observations with no missing values
    - flag: Integer to plot the PCA explained variance per PC (1) or not (0)

    Returns:
    --------
    - PCs_df: Pandas DataFrame containing the principal components data
    - loadings_df: Dictionary of the PCA loadings (correlation between PCs & stations data)

    """

    # In case the values are 0 it will complain about trying to divide by 0 (do not show warning)
    np.seterr(divide='ignore', invalid='ignore')

    # Initialize dictionary to save the loadings
    loadings_dict = {}

    # check that there are at least 2 years of data for this month
    if len(stations_data) > 1:

        # TEST
        # norm_stations_data = StandardScaler().fit_transform(stations_data.values[1::,:])

        # standardize data
        norm_stations_data = StandardScaler().fit_transform(stations_data.values)

        # run PCA
        pca = PCA()
        PCs_data = pca.fit_transform(norm_stations_data)

        # plt.plot(np.arange(len(PCs_data)),norm_stations_data)
        # plt.plot(np.arange(len(PCs_data)),PCs_data[:,0])
        # # plt.plot([-3,1],[-3,1])
        # plt.savefig("/Users/lla068/Desktop/test.png")
        # plt.close()

        # store PCs Numpy Array into a Pandas DataFrame
        columns = []
        for pc in range(PCs_data.shape[1]):
            columns.append('PC'+str(pc+1))
        PCs_df = pd.DataFrame(data=PCs_data, index=stations_data.index, columns=columns)

        # Plot the cumulative explained variance for all PCs
        if flag == 1:
            fig = plt.figure(figsize=(5,4))
            plt.plot(np.arange(1,len(pca.explained_variance_ratio_)+1),pca.explained_variance_ratio_, color='b', marker='o')
            plt.xticks(np.arange(1,len(pca.explained_variance_ratio_)+1))
            plt.ylim(0,1)
            plt.xlabel('PCs')
            plt.ylabel('explained variance');

        # combine stations and PCs data and calculate loadings (correlation between PCs & stations data)
        combined_stations_PCs_data = pd.concat([PCs_df, stations_data], axis=1)
        corr = combined_stations_PCs_data.corr(method='pearson') ** 2
        PCs_rows = [x for x in corr.columns if 'PC' in x]
        stations_cols = [x for x in corr.columns if 'PC' not in x]
        loadings_df = corr.loc[PCs_rows,stations_cols].dropna(axis=1)

        if flag == 0:
            return PCs_df, loadings_df
        elif flag == 1:
            return PCs_df, loadings_df, fig

    # Otherwise simply uses standardized SWE without PCA for this month
    elif len(stations_data) == 1:

        # store PC Numpy Array into a Pandas DataFrame
        PCs_df = pd.DataFrame(data=norm_stations_data, index=stations_data.index, columns=['PC1'])

        return PCs_df, None, None

    # Otherwise return nothing
    else:
        return None, None, None

###

def prob_metrics_calculation(Qobs, Qfc_ens, flag, niterations, perc_event_low, perc_event_high, min_obs, bins_thresholds):

    """Calculates deterministic metrics for whole hindcast timeseries (1 value per hindcast start date & target period).

    Keyword arguments:
    ------------------
    - Qobs: xarray Dataset containing timeseries of observed flow values for various target periods of flow accumulation.
    - Qfc_ens: xarray Dataset containing timeseries of ensemble flow forecasts for various target periods and forecast start dates.
    - flag: Integer to indicate whether the metrics should be calculated without (0) or with (1) bootstrapping
    - niterations: Integer > 0 of the number of bootstrapping iterations to perform
    - perc_event_low: Float between 0 and 1 to indicate the percentile of the low flow event for which calculations are made.
    - perc_event_high: Float between 0 and 1 to indicate the percentile of the high flow event for which calculations are made.
    - min_obs: Positive integer for the number of minimum number of observation-hindcast pairs required to calculate the various metrics
    - bins_thresholds: Numpy array of increasing probability thresholds to make the yes/no decision

    Returns:
    --------
    - crps_da: xarray DataArray containing the CRPS for each hindcast start date & target period.
    - crpss_da: xarray DataArray containing the CRPSS for each hindcast start date & target period.
    - reli_da: xarray DataArray containing the reliability index for each hindcast start date & target period.
    - roc_auc_da: xarray DataArray containing the ROC area under the curve for each hindcast start date & target period.
    - roc_da: xarray DataArray containing the ROC curves for each hindcast start date & target period.

    """

    # Read the start dates & target periods we will be calculating metrics for
    initdates = Qfc_ens.init_date.values
    targetperiods = list(Qfc_ens.keys())

    # Initialize the verification metrics' Numpy arrays
    if flag == 0:
        crpss_array = np.ones((len(initdates),len(targetperiods))) * np.nan
        reli_array = np.ones((len(initdates),len(targetperiods))) * np.nan
        roc_auc_array = np.ones((len(initdates),len(targetperiods),2)) * np.nan
        roc_array = np.ones((len(initdates),len(targetperiods),2,11,2)) * np.nan
    elif flag == 1:
        crpss_array = np.ones((len(initdates),len(targetperiods),niterations)) * np.nan
        reli_array = np.ones((len(initdates),len(targetperiods),niterations)) * np.nan
        roc_auc_array = np.ones((len(initdates),len(targetperiods),niterations,2)) * np.nan
        roc_array = np.ones((len(initdates),len(targetperiods),niterations,2,11,2)) * np.nan

    # Initialize the row counter to save results in the right place
    row = -1

    # Loop over forecast starting dates
    for i in initdates:

        # increment/initialize the row/column counters to save results in the right place
        row += 1
        column = -1

        # loop over target periods
        for t in targetperiods:

            # increment the column counter to save results in the right place
            column += 1

            # read forecasts & observations
            Qfc_ens_data = Qfc_ens[t].sel(init_date=i).dropna(dim='year')
            Qobs_data = Qobs[t].sel(year=Qfc_ens_data.year)

            # if there are no forecasts we skip this start date-target period
            if np.isnan(Qfc_ens_data.values).all() == True:
                continue

            # else, we can proceed with the verification metrics calculations
            else:

                if flag == 0:
                    # CRPS & CRPSS
                    crps_outputs = continuous_rank_prob_score(Qobs_data, Qfc_ens_data, min_obs)
                    crpss_array[row,column] = round(crps_outputs[1],2)
                    # Reliability index
                    reli_array[row,column] = round(reli_index(Qobs_data, Qfc_ens_data, min_obs),2)
                    # ROC
                    roc_outputs_low = ROC(Qobs_data, Qfc_ens_data, perc_event_low, 'infeq', min_obs, bins_thresholds)
                    roc_auc_array[row,column,0] = round(roc_outputs_low[1],2)
                    roc_array[row,column,:,:,0] = [roc_outputs_low[0].FAR.values, roc_outputs_low[0].HR.values]
                    roc_outputs_high = ROC(Qobs_data, Qfc_ens_data, perc_event_high, 'supeq', min_obs, bins_thresholds)
                    roc_auc_array[row,column,1] = round(roc_outputs_high[1],2)
                    roc_array[row,column,:,:,1] = [roc_outputs_high[0].FAR.values, roc_outputs_high[0].HR.values]

                elif flag == 1:

                    for b in range(niterations):

                        # random samples for bootstrapping with replacement
                        samples = np.random.choice(len(Qobs_data),len(Qobs_data))
                        Qfc_ens_data_bs = Qfc_ens_data.isel(year=samples)
                        Qobs_data_bs = Qobs_data.isel(year=samples)

                        # CRPS & CRPSS
                        crps_outputs = continuous_rank_prob_score(Qobs_data_bs, Qfc_ens_data_bs, min_obs)
                        crpss_array[row,column,b] = round(crps_outputs[1],2)
                        # Reliability index
                        reli_array[row,column,b] = round(reli_index(Qobs_data_bs, Qfc_ens_data_bs, min_obs),2)
                        # ROC
                        roc_outputs_low = ROC(Qobs_data_bs, Qfc_ens_data_bs, perc_event_low, 'infeq', min_obs, bins_thresholds)
                        roc_auc_array[row,column,b,0] = round(roc_outputs_low[1],2)
                        roc_array[row,column,b,:,:,0] = [roc_outputs_low[0].FAR.values, roc_outputs_low[0].HR.values]
                        roc_outputs_high = ROC(Qobs_data_bs, Qfc_ens_data_bs, perc_event_high, 'supeq', min_obs, bins_thresholds)
                        roc_auc_array[row,column,b,1] = round(roc_outputs_high[1],2)
                        roc_array[row,column,b,:,:,1] = [roc_outputs_high[0].FAR.values, roc_outputs_high[0].HR.values]

    # Save values to xarray DataArrays
    if flag == 0:
        crpss_da = xr.DataArray(data=crpss_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods]}, dims=['init_month','target_period'], name='CRPSS')
        reli_da = xr.DataArray(data=reli_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods]}, dims=['init_month','target_period'], name='Reliability_index')
        roc_auc_da = xr.DataArray(data=roc_auc_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods],'event':[perc_event_low, perc_event_high]}, dims=['init_date','target_period','event'], name='ROC_AUC')
        roc_da = xr.DataArray(data=roc_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods],'rate':['FAR','HR'],'bins':roc_outputs_high[0].bins,'event':[perc_event_low, perc_event_high]}, dims=['init_month','target_period','rate','bins','event'], name='ROC')

    elif flag == 1:
        crpss_da = xr.DataArray(data=crpss_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods],'iteration':np.arange(1,niterations+1)}, dims=['init_date','target_period','iteration'], name='CRPSS')
        reli_da = xr.DataArray(data=reli_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods],'iteration':np.arange(1,niterations+1)}, dims=['init_date','target_period','iteration'], name='Reliability_index')
        roc_auc_da = xr.DataArray(data=roc_auc_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods],'iteration':np.arange(1,niterations+1),'event':[perc_event_low, perc_event_high]}, dims=['init_date','target_period','iteration','event'], name='ROC_AUC')
        roc_da = xr.DataArray(data=roc_array, coords={'init_date':initdates,'target_period':[x[4::] for x in targetperiods],'iteration':np.arange(1,niterations+1),'rate':['FAR','HR'],'bins':roc_outputs_high[0].bins,'event':[perc_event_low, perc_event_high]}, dims=['init_date','target_period','iteration','rate','bins','event'], name='ROC')


    # Information for the output xarray DataArrays
    da_dict = {'CRPSS':crpss_da,'reli':reli_da,'ROC_AUC':roc_auc_da,'ROC':roc_da}
    metrics_longnames_dict = {'CRPSS':'Continuous Rank Probability Skill Score',
                              'reli':'Reliability index',
                              'ROC_AUC':'Relative Operating Characteristic (ROC) area under the curve (AUC)',
                              'ROC':'Relative Operating Characteristic (ROC)'}
    metrics_info_dict = {'CRPSS':'Measures the skill of the hindcast against a baseline (observations climatology). Range: -Inf to 1. Perfect score: 1. Units: Unitless.',
                         'reli':'Measures the closeness between the empirical CDF of the ensemble hindcast with the CDF of a uniform distribution (i.e., flat rank histogram). Range: 0 to 1. Perfect score: 1. Units: Unitless.',
                         'ROC_AUC':'Measures the ensemble hindcast resolution, its ability to discriminate between events (given percentile) & non-events. ROC AUC range: 0 to 1,. Perfect score: 1. No skill: 0.5. Units: Unitless.',
                         'ROC':'Measures the ensemble hindcast resolution, its ability to discriminate between events (given percentile) & non-events. The ROC curve plots the hit rate (HR) vs the false alarm rate (FAR) using a set of increasing probability thresholds (i.e., 0.1, 0.2, ..., 1) to make the yes/no decision.'}

    if flag == 1:
        for keys in da_dict.keys():
            da_dict[keys].iteration.attrs['long_name'] = 'Bootstrapping iterations'
            da_dict[keys].iteration.attrs['info'] = 'We pick random samples of the datasets for bootstrapping with replacement in order to capture uncertainties in the verification metric estimates.'

    # Add information to output xarray DataArrays
    for keys in da_dict.keys():
        da_dict[keys].init_date.attrs['long_name'] = 'Hindcast initialization date'
        da_dict[keys].init_date.attrs['info'] = 'DD/MM of the predictors used to generate the hindcasts.'
        da_dict[keys].target_period.attrs['long_name'] = 'Hindcast target period'
        da_dict[keys].target_period.attrs['info'] = 'Time period for which volumes are being forecasted.'
        da_dict[keys].attrs['long_name'] = metrics_longnames_dict[keys]
        da_dict[keys].attrs['info'] = metrics_info_dict[keys]

        if keys == 'ROC_AUC':
            da_dict[keys].event.attrs['info'] = 'The low flow & high flow events percentiles used to calculate the ROC, respectively. For low flow events, flows <= the observed low percentile are used. For high flow events, flows >= the observed high percentile are used.'
        if keys == 'ROC':
            da_dict[keys].event.attrs['info'] = 'The low flow & high flow events percentiles used to calculate the ROC, respectively. For low flow events, flows <= the observed low percentile are used. For high flow events, flows >= the observed high percentile are used.'
            da_dict[keys].bins.attrs['info'] = 'Forecast probability thresholds used for the ROC calculations.'
            da_dict[keys].rate.attrs['info'] = 'The false alarm rate (FAR) captures when an event is forecast to occur, but did not occur. The hite rate (HR) captures when an event is forecast to occur, and did occur.'

    return crpss_da, reli_da, roc_auc_da, roc_da

###

def qm_gap_filling(original_data, window_days, min_obs_corr, min_obs_cdf, min_corr):

    """Performing the gap filling for all missing observations (when possible) using quantile mapping.
    For each target station and each date for which date is missing, we identify a donor stations as the station with:
    - data for this date,
    - a cdf for this doy,
    - and the best correlation to the target station (correlation >= min_corr for this doy).

    Keyword arguments:
    ------------------
    - original_data: Pandas DataFrame of original stations' observations dataset, which will be gap filled
    - window_days: Positive integer denoting the number of days to select data for around a certain doy, to calculate correlations
    - min_obs_corr: Positive integer for the minimum number of overlapping observations required to calculate the correlation between 2 stations
    - min_obs_cdf: Positive integer for the minimum number of stations required to calculate a station's cdf
    - min_corr: Value between 0 and 1 for the minimum correlation value required to keep a donor station

    Returns:
    --------
    - gapfilled_data: Pandas DataFrame of gap filled stations' observations
    - data_type_flags: Pandas DataFrame with information about the type of data (estimates or observations) in the gap filled dataset
    - donor_stationIDs: Pandas DataFrame with information about the donor station used to fill each of the gaps

    """

    # Create a duplicate of the dataset to gap fill
    gapfilled_data = original_data.copy()

    # Remove P & external SWE stations (buffer) from the dataframe
    cols = [c for c in original_data.columns if 'precip' not in c and 'ext' not in c]

    # Keep only gap filled SWE stations (without P stations & external SWE stations)
    gapfilled_data = gapfilled_data[cols]

    # Add doy to the Pandas DataFrame
    original_data['doy'] = original_data.index.dayofyear

    # Set empty dataframes to keep track of data type and donor station ids
    data_type_flags = pd.DataFrame(data=0, index=original_data.index, columns=cols)
    donor_stationIDs = pd.DataFrame(data=np.nan, index=original_data.index, columns=cols)

    # Calculate correlations between stations that have overlapping observations
    corr = calculate_stations_doy_corr(original_data, window_days, min_obs_corr)

    # Identify dates for gap filling
    time_index = original_data.index

    # Loop over dates for gap filling
    for d in time_index:

        # Calculate the doy corresponding to the date
        # Note: doy 365 and 366 are bundled together
        doy = original_data.loc[d,'doy']

        # Calculate the start and end dates of the time window for the gap filling steps
        window_startdate = d - pd.Timedelta(days=window_days)
        window_enddate = d + pd.Timedelta(days=window_days)

        # Get IDs of all stations with data for this date (and within time window)
        data_window = original_data[window_startdate:window_enddate].dropna(axis=1, how='all')
        non_missing_stations = [c for c in data_window.columns if 'doy' not in c]
        data_window['days_to_date'] = abs((d - data_window.index).days)

        # Calculate the start & end doys of the time window for quantile mapping, with special rules around the start & end of the calendar year
        window_startdoy = (data_window['doy'].iloc[0])%366
        window_startdoy = 366 if window_startdoy == 0 else window_startdoy
        window_enddoy = (data_window['doy'].iloc[-1])%366
        window_enddoy = 366 if window_enddoy == 0 else window_enddoy

        # Loop over stations to gap fill
        for target_station in cols:

            # If station has no data, proceed with the gap filling
            if np.isnan(original_data.loc[d,target_station]):

                # Select target data within time window for this doy from all years
                if window_startdoy > window_enddoy:
                    data_window_target = original_data[target_station].dropna()[(original_data['doy']>=window_startdoy) | (original_data['doy'] <= window_enddoy)]
                else:
                    data_window_target = original_data[target_station].dropna()[(original_data['doy']>=window_startdoy) & (original_data['doy'] <= window_enddoy)]

                # We can continue if there are enough target data to build cdf
                if len(data_window_target.index) >= min_obs_cdf:

                    # Get ids of all stations with correlations >= a minimum correlation for this doy, not including the target station itself
                    non_missing_corr = corr[doy][target_station].dropna()
                    non_missing_corr = non_missing_corr[non_missing_corr.index.isin(non_missing_stations)]
                    potential_donor_stations = non_missing_corr[non_missing_corr >= min_corr].index.values
                    potential_donor_stations = [c for c in potential_donor_stations if target_station not in c]

                    # If there is at least one potential donor station, proceed
                    if len(potential_donor_stations) > 0:

                        # Sort the donor stations from highest to lowest value
                        potential_donor_stations_sorted = corr[doy].loc[potential_donor_stations,target_station].dropna().sort_values(ascending=False).index.values

                        # Loop over sorted donor stations until I find one with enough data to build a cdf
                        for donor_station in potential_donor_stations_sorted:

                            # Select data within time window for this doy from all years
                            if window_startdoy > window_enddoy:
                                data_window_donor = original_data[donor_station].dropna()[(original_data['doy'] >= window_startdoy) | (original_data['doy'] <= window_enddoy)]
                            else:
                                data_window_donor = original_data[donor_station].dropna()[(original_data['doy'] >= window_startdoy) & (original_data['doy'] <= window_enddoy)]

                            # We can continue if there are enough donor data to build cdf
                            if len(data_window_donor.index) >= min_obs_cdf:

                                # If the donor station has multiple values within the window, we keep the closest donor station value to the date we are gap filling
                                sorted_data_window = data_window.sort_values(by=['days_to_date'])
                                value_donor = sorted_data_window[donor_station].dropna()[0]

                                # Perform the gap filling using quantile mapping
                                value_target = quantile_mapping(data_window_donor, data_window_target, value_donor, min_obs_cdf, flag=0)

                                if value_target != None:
                                    gapfilled_data.loc[d,target_station] = value_target
                                    data_type_flags.loc[d,target_station] = 1
                                    donor_stationIDs.loc[d,target_station] = donor_station

                                break

                            else:
                                continue

    return gapfilled_data, data_type_flags, donor_stationIDs

###

def quantile_mapping(data_donor, data_target, value_donor, min_obs_cdf, flag):

    """Calculating target station's gap filling value from donor station's value using quantile mapping.

    Keyword arguments:
    ------------------
    - data_donor: Pandas DataFrame of donor station observations used to build empirical cdf
    - data_target: Pandas DataFrame of target station observations used to build empirical cdf
    - value_donor: Integer of donor station value used in the quantile mapping
    - min_obs_cdf: Positive integer for the minimum number of unique observations required to calculate a station's cdf
    - flag: Integer to plot (1) or not (0) the donor and target stations' cdfs

    Returns:
    --------
    - value_target: Integer of target station value calculated using quantile mapping
    - plot of the donor and target stations' cdfs (optional)

    """

    # build the donor station's empirical cdf
    sorted_data_donor = data_donor.drop_duplicates().sort_values(ignore_index=True)

    # build the target station's empiral cdf
    sorted_data_target = data_target.drop_duplicates().sort_values(ignore_index=True)

    # Calculate the donor & target stations' cdfs if they both have at least X unique observations
    if (len(sorted_data_donor) >= min_obs_cdf) & (len(sorted_data_target) >= min_obs_cdf):

        # Calculate the cumulative probability corresponding to the donor value
        rank_donor_obs = sorted_data_donor[sorted_data_donor == value_donor].index[0]
        total_obs_donor = len(sorted_data_donor)
        cumul_prob_donor_obs = (rank_donor_obs + 1) / total_obs_donor

        # Calculate the cumulative probability corresponding to the target value
        cumul_prob_target = np.arange(1,len(sorted_data_target)+1) / (len(sorted_data_target))

        # inter-/extrapolate linearly to get the target value corresponding to the donor station's cumulative probability
        inverted_edf = interp1d(cumul_prob_target, sorted_data_target, fill_value="extrapolate")
        value_target = round(float(inverted_edf(cumul_prob_donor_obs)),2)

        # set any potential negative values from interpolation/extrapolation to zero
        if(value_target) < 0:
            value_target = 0

        # if requested, plot the target & donor stations' cdfs
        if flag == 1:
            plt.figure()
            plt.plot(sorted_data_donor, np.arange(1,len(sorted_data_donor)+1) / (len(sorted_data_donor)), label='donor')
            plt.plot(sorted_data_target, cumul_prob_target, label='target')
            plt.scatter(value_donor, cumul_prob_donor_obs)
            plt.legend()

        return value_target

    # If either/both the target & donor stations have < X observations do nothing
    else:
        return None

###

def regime_classification(streamflow_obs, start_water_year, max_gap_days, flag):

    """Performs the regime classification for a given method (i.e., flag), using circular statistics from Burn et al. (2010): https://doi.org/10.1002/hyp.7625

    Keyword arguments:
    ------------------
    - streamflow_obs: xarray Dataset of streamflow observations
    - start_water_year: Tuple with (month, day) of the water year starting date
    - max_gap_days: Positive integer of the max. number of days for gaps allowed in the daily streamflow data for the linear interpolation
    - flag: An integer of 1, 2 or 3 defining the method to be used for identifying streamflow peaks
      -> flag=1: streamflow annual maxima
      -> flag=2: peak over threshold (POT) where the threshold = minimum value of all annual maxima
      -> flag=3: annual centres of mass (i.e., doy where 1/2 of the total water year streamflow has passed through the river - see: https://journals.ametsoc.org/view/journals/clim/18/2/jcli-3272.1.xml)

    Returns:
    --------
    - basins_regimes_gdf: Pandas GeoDataFrame containing station information (e.g., outlet lat, lon and ID) & circular statistics for all stations

    """

    # Set empty lists to store information on stations & their circular statistics
    theta_rad_stations = []
    regularity_stations = []
    doy_stations = []

    # extract all stations for plotting
    stations_to_plot = [x for x in streamflow_obs.Station_ID.values]

    # Create Pandas GeoDataFrame to save circular statistics
    basins_regimes_df = streamflow_obs.Station_ID.to_dataframe().reset_index(drop=True)
    basins_regimes_gdf = gpd.GeoDataFrame(basins_regimes_df, geometry=gpd.points_from_xy(basins_regimes_df.lon, basins_regimes_df.lat))
    basins_regimes_gdf = basins_regimes_gdf.drop(columns=['lat','lon'])
    basins_regimes_gdf.crs = {"init":"epsg:4326"}

    # Empty list to store stations without stats so we can remove them from the final output
    stations_without_stats = []

    # set counter
    n = -1

    # loop over the stations
    for s in stations_to_plot:

        n += 1

        # read streamflow observations for basin as xarray DataArray
        if len(stations_to_plot) == 1:
            streamflow_data_da = streamflow_obs.Flow
        else:
            streamflow_data_da = streamflow_obs.sel(Station_ID=s).Flow

        # run linear interpolation to maximize the number of values available to calculate the peak flow statistcs and save interpolated outputs to pandas DataFrame
        streamflow_data_da = streamflow_data_da.interpolate_na(method='linear', dim='time', max_gap=datetime.timedelta(days=max_gap_days))
        streamflow_data_df = streamflow_data_da.to_dataframe()
        streamflow_data_df.reset_index(inplace=True)
        streamflow_data_df['year'] = streamflow_data_df.time.map(lambda x: x.year)
        streamflow_data_df['doy'] = streamflow_data_df.time.map(lambda x: x.dayofyear)
        streamflow_data_df = streamflow_data_df.set_index('doy')

        # add a water year column
        water_year = []
        for i in pd.DatetimeIndex(streamflow_data_df.time.values):
            if (i.month == start_water_year[0] and i.day >= start_water_year[1]) or (i.month > start_water_year[0]):
                water_year.append(i.year + 1)
            else:
                water_year.append(i.year)
        streamflow_data_df['water_year'] = water_year

        # calculate the basin's streamflow statistics for the chosen method
        streamflow_stats = streamflow_peaks_statistics(streamflow_data_df, flag)

        # if we have peak flow data, calculate the basin's circular statistics
        if streamflow_stats.empty == False:
            theta_rad, regularity = circular_stats(streamflow_stats.doy.values, streamflow_stats.length_year.values)

            # save the circular statistics & basin's information
            theta_rad_stations.append(circmean(theta_rad))
            regularity_stations.append(regularity)
            doy_stations.append(circmean(streamflow_stats.doy.values, high=366, low=1))

        # if not, output NaNs
        else:
            print("Station "+s+" doesn't have any complete years of streamflow data to calculate peak statistics.")

            # save the circular statistics & basin's information
            theta_rad_stations.append(np.nan)
            regularity_stations.append(np.nan)
            doy_stations.append(np.nan)

    # Append circular statistics to Pandas GeoDataFrame
    basins_regimes_gdf['circular_stats_theta_rad'] = theta_rad_stations
    basins_regimes_gdf['circular_stats_regularity'] = regularity_stations
    basins_regimes_gdf['mean_peak_doy'] = doy_stations

    return basins_regimes_gdf, theta_rad

###

def reli_index(Qobs, Qfc_ens, min_obs):

    """Calculates the reliability index.
    A measure of the average agreement between the predictive distribution (from the ensemble forecasts) & the observations.
    Quantifies the closeness between the empirical CDF of the forecast with the CDF of a uniform distribution (i.e., flat rank histogram).
    Range: 0 to 1. Perfect score: 1. Units: Unitless.
    For more info, see Renard et al. (2010): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009WR008328 & Mendoza et al. (2017): https://hess.copernicus.org/articles/21/3915/2017/hess-21-3915-2017.pdf

    Keyword arguments:
    ------------------
    - Qobs: xarray DataArray containing a timeseries of observed flow values.
    - Qfc_ens: xarray DataArray containing a timeseries of ensemble flow forecasts.
    - min_obs: Positive integer for the number of minimum number of observation-hindcast pairs required to calculate the various metrics

    Returns:
    --------
    - alpha: Reliability index value between the ensemble forecast and a uniform distribution.

    """

    if len(Qobs) >= min_obs:

        # Transform ensemble forecast xarray DataArray into a Pandas dataframe
        Qfc_obs = pd.DataFrame(data=Qfc_ens.values, index=Qfc_ens.year, columns=Qfc_ens.ens_member)

        # Combine forecasts and observations
        Qfc_obs['obs'] = Qobs.values

        # Rank dataframe to find the position of observation within the ensemble forecast for each year
        Qfc_obs_ranked = Qfc_obs.rank(axis=1)

        # Create histogram of position of observations in each bin (with nbin=ensemble size+1)
        Qobs_hist = np.histogram(Qfc_obs_ranked['obs'].values, bins=len(Qfc_obs.columns))[0] / len(Qfc_obs.index)

        # Accumulate the histogram
        Qobs_hist_cumul = np.cumsum(Qobs_hist)

        # Uniform distribution from 0 to 1
        uniform_dist = np.linspace(1/len(Qfc_obs.columns),1,len(Qfc_obs.columns))

        # Calculate the reliability index
        alpha = 1 - 2 * np.mean(abs(Qobs_hist_cumul - uniform_dist))

    else:

        alpha = np.nan

    return alpha

###

def ROC(Qobs, Qfc_ens, percentile, sign, min_obs, bins_thresholds):

    """Function to calculate the Relative Operating Characteristic (ROC) for a given percentile.
    It measures the ability of the forecast to discriminate between events (given percentile) and non-events and says something about its resolution.
    It is the equivalent of the contingency table for deterministic forecasts.
    The ROC can thus be considered as a measure of potential usefulness.
    The ROC is conditioned on the observations (i.e., given that Y occurred, what was the correponding forecast?). It is therefore a good companion to a reliability metric, which would be conditioned on the forecasts.
    It is not sensitive to bias in the forecast, so says nothing about reliability. A biased forecast may still have good resolution, and it may be possible to improve the forecast through calibration.
    The ROC curve plots the hit rate vs the false alarm rate using a set of increasing probability thresholds (e.g., 0.1, 0.2, ..., 1) to make the yes/no decision.
    A perfect forecast has a curve that travels from bottom left to top left of diagram, then across to top right of diagram. A diagonal line indicates no skill.
    The area under the ROC curve (ROC AUC) can be used as a score to summarize the ROC curve information.
    ROC AUC range: 0 to 1,. Perfect score: 1. No skill: 0.5. Units: Unitless.
    For more info, see Mason (1982): http://www.bom.gov.au/jshess/docs/1982/mason.pdf & the sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html

    Keyword arguments:
    ------------------
    - Qobs: xarray DataArray containing a timeseries of observed flow values.
    - Qfc_ens: xarray DataArray containing a timeseries of ensemble flow forecasts.
    - percentile: Float between 0 and 1 to indicate the percentile of the event for which calculations are made (e.g., 0.5 will mean that we look at flows either below or above the median of all observations).
    - sign: String indicating the side of the percentile to use as a threshold for calculations. It can be 'supeq' for >= given percentile or 'infeq' for <= percentile.
    - min_obs: Positive integer for the number of minimum number of observation-hindcast pairs required to calculate the various metrics
    - bins_thresholds: Numpy array of increasing probability thresholds to make the yes/no decision

    Returns:
    --------
    - roc_curve: Pandas DataFrame containing the ROC curve information containing the false alarm rate & hit rate for bins of the data.
    - roc_auc: ROC area under the curve value.

    """

    if len(Qobs) >= min_obs:

        # Calculate the observed flow threshold = event
        threshold = Qobs.quantile(q=percentile)

        # Calculate the forecast probability to fall above/below this threshold and translating into bins for the reliability table
        if sign == 'supeq':
            Qfc_prob = (threshold <= Qfc_ens).sum(axis=1) / Qfc_ens.shape[1]
        elif sign == 'infeq':
            Qfc_prob = (threshold >= Qfc_ens).sum(axis=1) / Qfc_ens.shape[1]

        # Check which observations fall above/below this threshold and saving as binary information (1: yes, 0: no)
        if sign == 'supeq':
            Qobs_bin = (threshold <= Qobs).astype(dtype=int)
        elif sign == 'infeq':
            Qobs_bin = (threshold >= Qobs).astype(dtype=int)

        # Calculate ROC
        hr_list = []
        far_list = []

        for t in bins_thresholds:
            y_class = Qfc_prob.values >= t

            # calculate components of contingency table
            TP = sum((Qobs_bin.values == 1) & (y_class == 1))
            FN = sum((Qobs_bin.values == 1) & (y_class == 0))
            FP = sum((Qobs_bin.values == 0) & (y_class == 1))
            TN = sum((Qobs_bin.values == 0) & (y_class == 0))

            # calculate hit rate and false alarm rate
            if TP == 0:
                hr = 0
            else:
                hr = TP / (TP + FN)
            if FP == 0:
                far = 0
            else:
                far = FP / (FP + TN)
            hr_list.append(hr)
            far_list.append(far)

        # Calculate ROC area under curve (AUC)
        roc_auc = metrics.auc(far_list, hr_list)

        # Save information to df
        data = {'bins': bins_thresholds, 'FAR': far_list, 'HR': hr_list}
        roc_curve = pd.DataFrame(data=data, columns=['bins', 'FAR', 'HR'])

    else:

        data = {'bins': bins_thresholds, 'FAR': [np.nan]*len(bins_thresholds), 'HR': [np.nan]*len(bins_thresholds)}
        roc_curve = pd.DataFrame(data=data, columns=['bins', 'FAR', 'HR'])
        roc_auc = np.nan

    return roc_curve, roc_auc

###

def ROC_plots(metric_values, percentile):

    """Plots ROC curves for a given event (i.e., percentile).

    Keyword arguments:
    ------------------
    - metric_values: xarray DataArray containing verification metric values for various target periods and forecast start dates.
    - percentile: Float between 0 and 1 to indicate the percentile of the event for which calculations are made (e.g., 0.5 will mean that we look at flows either below or above the median of all observations).

    Returns:
    --------
    - Sub-plots of the ROC curves for each target period (start dates are represented on the same sub-plot for the same target period).

    """

    # Initialize figure
    nrows = int(len(metric_values.target_period)/3+len(metric_values.target_period)%3)
    fig, ax = plt.subplots(3,nrows, figsize=[10,10])

    # Initialize counter to create subplots in the right place
    elem = -1

    # Set color depending on the event
    if percentile < 0.5:
        color='orange'
    else:
        color='b'

    # Loop over forecast target periods
    for t in metric_values.target_period.values:

        # increment counter
        elem += 1

        # loop over forecast start dates to plot ROC curves
        for i in metric_values.init_month.values:
            ax[int(np.floor(elem/3)),elem%3].plot(metric_values.sel(target_period=t, init_month=i, event=percentile, rate='FAR'), metric_values.sel(target_period=t, init_month=i, event=percentile, rate='HR'), color=color, marker='o')

        # plot no skill line
        ax[int(np.floor(elem/3)),elem%3].plot([0,1], [0,1], color='grey', ls='--', lw=.5)

        # Add labels & legend
        ax[int(np.floor(elem/3)),elem%3].set_title(t)
        if int(np.floor(elem/3)) == nrows-1:
            ax[int(np.floor(elem/3)),elem%3].set_xlabel('False Alarm Rate')
            if elem%3 == 0:
                ax[int(np.floor(elem/3)),elem%3].set_ylabel('Hit Rate')
        else:
            ax[int(np.floor(elem/3)),elem%3].set_xticklabels([])
        if elem%3 != 0:
            ax[int(np.floor(elem/3)),elem%3].set_yticklabels([]);

###

def split_sample(original_timeseries):

    """Splits predictor & predictand timeseries data in half for split sample testing (training & testing the forecast model).

    Keyword arguments:
    ------------------
    - original_timeseries: Pandas DataFrame of the combined predictor & predictand timeseries

    Returns:
    --------
    - train_timeseries: Pandas DataFrame containing the 1st half of the timeseries used for training the forecast model
    - test_timeseries: andas DataFrame containing the 2nd half of the timeseries used for testing the forecast model

    """

    split_point = int(len(original_timeseries.index)/2)
    train_timeseries = original_timeseries[0:split_point]
    test_timeseries = original_timeseries[split_point:]

    return train_timeseries, test_timeseries

###

def stations_basin_map(basins, basin_id, SWE_stations, P_stations, flag, buffer_km=0):

    """Plots map of SWE and P stations in and around the basin.

    Keyword arguments:
    ------------------
    - basins: Pandas GeoDataFrame of all basins available
    - basin_id: String of the basin id to plot
    - SWE_stations: Pandas GeoDataFrame of SWE stations to plot
    - P_stations: Pandas GeoDataFrame of P stations to plot
    - dem_dir: String of the path to DEMs
    - flag: Flag to indicate if no buffer (0) or a buffer (1) should be plotted around the basin
    - buffer_km: Positive value (in km) of buffer to add around basin shapefile (default=0; i.e., no buffer)

    Returns:
    --------
    - A map of SWE stations and basin shape.

    """

    # Initialize the figure
    fig, ax = plt.subplots()

    # If there is no buffer
    if flag == 0:
        # calculating map bounds
        minx, miny, maxx, maxy = basins.loc[basins.Station_ID == basin_id].geometry.total_bounds

    # If there is a buffer
    if flag == 1:

        # convert basin geometry to a different epsg to be able to add a buffer in meters
        basin_crs_conversion = basins.loc[basins.Station_ID == basin_id].to_crs(epsg=3763)

        # add a buffer in meters
        buffer_m = buffer_km * 1000
        basin_buffer = basin_crs_conversion.buffer(buffer_m)

        # convert the buffer back to meters for plotting
        basin_buffer = basin_buffer.to_crs(epsg=4326)

        # calculate map bounds
        minx, miny, maxx, maxy = basin_buffer.geometry.total_bounds

        # plot the basin buffer
        basin_buffer.plot(ax=ax, alpha=.5)

    # Plot the basin and SWE stations
    # src = rasterio.open(dem_dir + basin_id[:2] + ".tif")
    # rasterio.plot.show((src, 1), cmap='Greys', vmin=0, ax=ax, alpha=.7)
    basins.loc[basins.Station_ID == basin_id].plot(ax=ax, alpha=.3)
    SWE_stations.plot(ax=ax, marker='o', color='b', markersize=10, alpha=.8, label='SWE stations')
    P_stations.plot(ax=ax, marker='o', color='g', markersize=10, alpha=.8, label='P stations')

    ax.legend()
    plt.title(basins.loc[basins.Station_ID == basin_id]['Station_Na'].values[0])
    ax.set_xlim(minx - .1, maxx + .1)
    ax.set_ylim(miny - .1, maxy + .1);

    return fig

###

def streamflow_peaks_statistics(streamflow_data, flag):

    """Identifies the streamflow peaks for a given method (i.e., flag).

    Keyword arguments:
    ------------------
    - streamflow_data: Pandas DataFrame of the daily streamflow observations for a basin
    - flag: An integer of 1, 2 or 3 defining the method to be used for identifying streamflow peaks
      -> flag=1: streamflow annual maxima
      -> flag=2: peak over threshold (POT) where the threshold = minimum value of all annual maxima
      -> flag=3: annual centres of mass (i.e., doy where 1/2 of the total water year streamflow has passed through the river - see: https://journals.ametsoc.org/view/journals/clim/18/2/jcli-3272.1.xml)

    Returns:
    --------
    - streamflow_stats: Pandas DataFrame of the streamflow peaks statistics

    """

    # Calculate streamflow annual maxima - we need the annual maxima to calculate the POT
    if flag == 1 or flag == 2:
        streamflow_data_peaks = streamflow_data.groupby("water_year").Flow.max().dropna(axis=0)
        streamflow_data_peaks_doys = streamflow_data.groupby("water_year").Flow.idxmax(skipna=False).dropna(axis=0)
        streamflow_stats = pd.DataFrame(data=streamflow_data_peaks_doys.values, index=streamflow_data_peaks_doys.index, columns=["doy"])
        dates = pd.to_datetime([str(x)+"-01-01" for x in streamflow_stats.index.values], format='%Y-%m-%d')
        leap_years = dates.is_leap_year.astype(int)
        streamflow_stats['length_year'] = [366 if x == 1 else 365 for x in leap_years]
        streamflow_stats['annual_maxima'] = streamflow_data_peaks[streamflow_data_peaks_doys.index]

    # Calculate peak over threshold (POT)
    if flag == 2:
        threshold = streamflow_stats['annual_maxima'].min() # threshold = minimum value of all annual maxima
        streamflow_data_peaks = streamflow_data[streamflow_data.Flow>threshold]
        streamflow_stats = pd.DataFrame(data=streamflow_data_peaks).drop(columns=['Station_ID','time','lat','lon']).reset_index()
        dates = pd.to_datetime([str(x)+"-01-01" for x in streamflow_stats.year.values], format='%Y-%m-%d')
        leap_years = dates.is_leap_year.astype(int)
        streamflow_stats['length_year'] = [366 if x == 1 else 365 for x in leap_years]

    # Centre of mass
    elif flag == 3:
        streamflow_data_peaks = []
        streamflow_data_peaks_doys = []
        water_years = []
        for wy in list(set(streamflow_data['water_year'].values)):
            streamflow_cumsum = streamflow_data[streamflow_data['water_year']==wy].Flow.cumsum(skipna=True)
            if streamflow_cumsum.isnull().values.any() == False:
                water_years.append(wy)
                mid_pt = streamflow_cumsum.values[-1]/2
                mid_pt_position = np.abs(streamflow_cumsum.values - mid_pt).argmin()
                centre_of_mass = streamflow_cumsum.iloc[mid_pt_position]
                streamflow_data_peaks.append(centre_of_mass)
                streamflow_data_peaks_doys.append(streamflow_cumsum[streamflow_cumsum==centre_of_mass].index.values[0])
            else:
                continue
        streamflow_stats = pd.DataFrame(data=streamflow_data_peaks_doys, index=np.arange(len(streamflow_data_peaks_doys)), columns=["doy"])
        dates = pd.to_datetime([str(x)+"-01-01" for x in water_years], format='%Y-%m-%d')
        leap_years = dates.is_leap_year.astype(int)
        streamflow_stats['length_year'] = [366 if x == 1 else 365 for x in leap_years]
        streamflow_stats['centre_of_mass'] = streamflow_data_peaks

    return streamflow_stats

###
