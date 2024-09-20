import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from siphon.catalog import TDSCatalog
import metpy.calc as mpcalc
from scipy.ndimage import gaussian_filter
from metpy.units import units

# Defined function to find the time dimension in the dataset
def time_dim(ds, var_name):
    possible_time_dims = ['time', 'time1', 'time2', 'time3']
    time_dim = None
    for dim in possible_time_dims:
        if dim in ds[var_name].dims:
            time_dim = dim
            break
    if time_dim is None:
        raise ValueError('Could not find the time dimension')
    return time_dim

# Defined function to extract the data from the GFS model
def function_a(directions):
    tds_gfs = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/latest.html')
    gfs_ds = tds_gfs.datasets[0]
    ds = xr.open_dataset(gfs_ds.access_urls['OPENDAP'])
    ds_sliced = ds.sel(lat=slice(directions['North'], directions['South']), lon=slice(directions['West'], directions['East']))
    u = ds_sliced['u-component_of_wind_isobaric'].sel(isobaric=slice(50000, 100000))
    v = ds_sliced['v-component_of_wind_isobaric'].sel(isobaric=slice(50000, 100000))
    z = ds_sliced['Geopotential_height_isobaric'].sel(isobaric=slice(50000, 100000))
    temp = ds_sliced['Temperature_isobaric'].sel(isobaric=slice(50000, 100000))
    mslp = ds_sliced['MSLP_Eta_model_reduction_msl']
    pressure_levels = u['isobaric'].values[::-1]
    time = time_dim(ds, 'u-component_of_wind_isobaric')
    return u, v, z, temp, mslp, pressure_levels, time


# Defined function to calculate and plot the temperature advection at 850 hPa
def function_b(g, u, v, z, temp, mslp, pressure_levels, directions, time):
    # Select the model initialization time 
    int_time = u[time][0].values
    
    # Loop through the forecast hours
    for i in range(0, 1): 
        # Select the variables at the current forecast hour
        u_i = u.isel(**{time: i})
        v_i = v.isel(**{time: i})
        z_i = z.isel(**{time: i})
        temp_i = temp.isel(**{time: i})
        mslp_i = mslp.isel(**{time: i})
        time_values = u_i[time].values
        
        # Select the variables at the pressure levels of interest
        u_ij = u_i.sel(isobaric=pressure_levels)
        v_ij = v_i.sel(isobaric=pressure_levels)
        z_ij = z_i.sel(isobaric=pressure_levels)
        temp_ij = temp_i.sel(isobaric=pressure_levels)
        
        #wnd_ij = mpcalc.wind_speed(u_ij, v_ij)

        # Select the variables at 850 hPa and multiply by the appropriate units
        temp_ij_850 = temp_ij.sel(isobaric=85000) * units.kelvin
        u_ij_850 = u_ij.sel(isobaric=85000) * units.meter / units.second
        v_ij_850 = v_ij.sel(isobaric=85000) * units.meter / units.second 
        z_ij_850 = z_ij.sel(isobaric=85000) * units.meter

        # Calculate temperature advection at 850 hPa
        tadv_850 = mpcalc.advection(temp_ij_850, v_ij_850, u_ij_850) # Units are Kevlin / second
        tadv_850_hr = tadv_850 * 3600 # Units are Kelvin / hour

        # Convert to an array 
        tadv_da = xr.DataArray(tadv_850_hr, dims=['lat', 'lon'], coords={'lat': u['lat'], 'lon': u['lon']}, name=f'tadv_time_{i}')
        print(tadv_da)

        # Use Pandas to convert the time to a datetime object for plotting
        int_datetime_index = pd.DatetimeIndex([int_time])
        datetime_index = pd.DatetimeIndex([time_values])
        time_diff_hours = int((datetime_index[0] - int_datetime_index[0]).total_seconds() / 3600)

        # Color map and normalization for temperature advection
        levels = np.arange(-6, 7, 1)
        cmap = plt.get_cmap('bwr')
        norm = mcolors.BoundaryNorm(levels, cmap.N)

        # Define levels for geopotential height contours
        reference_height = 1500 # Units in meters
        start_level = reference_height - 300
        end_level = reference_height + 300
        step_size = 30
        isohypses_levels = np.arange(start_level, end_level + step_size, step_size)

        # Make the plot 
        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot the statelines, etc. 
        ax.set_extent([directions['West'], directions['East'], directions['South'], directions['North']-1])
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5, zorder=1)
        ax.add_feature(cfeature.OCEAN, color='#ecf9fd', zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=1)
        ax.add_feature(cfeature.LAKES, color='#ecf9fd', zorder=1)
        ax.add_feature(cfeature.LAND, color='#fbf5e9', zorder=1)
        gls = ax.gridlines(draw_labels=True, color='black', linestyle='--', alpha=0.35, zorder=1)
        gls.top_labels = False
        gls.right_labels = False

        # Plot the data 
        c = plt.contour(z_ij_850['lon'], z_ij_850['lat'], gaussian_filter(z_ij_850, sigma=1), colors='black', levels=isohypses_levels)
        ax.clabel(c, levels=isohypses_levels, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        c = plt.contour(tadv_da['lon'], tadv_da['lat'], gaussian_filter(tadv_da, sigma=1), colors='black', levels=levels, linewidths=0.5)
        cf = plt.contourf(tadv_da['lon'], tadv_da['lat'], gaussian_filter(tadv_da, sigma=1), cmap=cmap, levels=levels, norm=norm, extend='both')
        plt.colorbar(cf, extend='max', orientation='vertical', label='C/hr', fraction=0.046, pad=0.04)
        plt.title(f'{int_datetime_index[0].strftime("%H00 UTC")} GFS 850 hPa Temperature Advection | {datetime_index[0].strftime("%Y-%m-%d %H00 UTC")} | FH: {time_diff_hours}', fontsize=14)
        plt.tight_layout()
        plt.show()

# Defined function to calculate and plot the temperature advection at 700 hPa
def function_c(g, u, v, z, temp, mslp, pressure_levels, directions, time):
    # Select the model initialization time 
    int_time = u[time][0].values
    
    # Loop through the forecast hours
    for i in range(0, 1): 
        # Select the variables at the current forecast hour
        u_i = u.isel(**{time: i})
        v_i = v.isel(**{time: i})
        z_i = z.isel(**{time: i})
        temp_i = temp.isel(**{time: i})
        mslp_i = mslp.isel(**{time: i})
        time_values = u_i[time].values
        
        # Select the variables at the pressure levels of interest
        u_ij = u_i.sel(isobaric=pressure_levels)
        v_ij = v_i.sel(isobaric=pressure_levels)
        z_ij = z_i.sel(isobaric=pressure_levels)
        temp_ij = temp_i.sel(isobaric=pressure_levels)
        
        #wnd_ij = mpcalc.wind_speed(u_ij, v_ij)

        # Select the variables at 700 hPa and multiply by the appropriate units
        temp_ij_700 = temp_ij.sel(isobaric=70000) * units.kelvin
        u_ij_700 = u_ij.sel(isobaric=70000) * units.meter / units.second
        v_ij_700 = v_ij.sel(isobaric=70000) * units.meter / units.second 
        z_ij_700 = z_ij.sel(isobaric=70000) * units.meter

        # Calculate temperature advection at 700 hPa
        tadv_700 = mpcalc.advection(temp_ij_700, v_ij_700, u_ij_700) # Units are Kevlin / second
        tadv_700_hr = tadv_700 * 3600 # Units are Kelvin / hour

        # Convert to an array 
        tadv_da = xr.DataArray(tadv_700_hr, dims=['lat', 'lon'], coords={'lat': u['lat'], 'lon': u['lon']}, name=f'tadv_time_{i}')
        print(tadv_da)

        # Use Pandas to convert the time to a datetime object for plotting
        int_datetime_index = pd.DatetimeIndex([int_time])
        datetime_index = pd.DatetimeIndex([time_values])
        time_diff_hours = int((datetime_index[0] - int_datetime_index[0]).total_seconds() / 3600)

        # Color map and normalization for temperature advection
        levels = np.arange(-6, 7, 1)
        cmap = plt.get_cmap('bwr')
        norm = mcolors.BoundaryNorm(levels, cmap.N)

        # Define levels for geopotential height contours
        reference_height = 3000 # Units in meters
        start_level = reference_height - 300
        end_level = reference_height + 300
        step_size = 30
        isohypses_levels = np.arange(start_level, end_level + step_size, step_size)

        # Make the plot 
        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot the statelines, etc. 
        ax.set_extent([directions['West'], directions['East'], directions['South'], directions['North']-1])
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5, zorder=1)
        ax.add_feature(cfeature.OCEAN, color='#ecf9fd', zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=1)
        ax.add_feature(cfeature.LAKES, color='#ecf9fd', zorder=1)
        ax.add_feature(cfeature.LAND, color='#fbf5e9', zorder=1)
        gls = ax.gridlines(draw_labels=True, color='black', linestyle='--', alpha=0.35, zorder=1)
        gls.top_labels = False
        gls.right_labels = False

        # Plot the data 
        c = plt.contour(z_ij_700['lon'], z_ij_700['lat'], gaussian_filter(z_ij_700, sigma=1), colors='black', levels=isohypses_levels)
        ax.clabel(c, levels=isohypses_levels, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        c = plt.contour(tadv_da['lon'], tadv_da['lat'], gaussian_filter(tadv_da, sigma=1), colors='black', levels=levels, linewidths=0.5)
        cf = plt.contourf(tadv_da['lon'], tadv_da['lat'], gaussian_filter(tadv_da, sigma=1), cmap=cmap, levels=levels, norm=norm, extend='both')
        plt.colorbar(cf, extend='max', orientation='vertical', label='C/hr', fraction=0.046, pad=0.04)
        plt.title(f'{int_datetime_index[0].strftime("%H00 UTC")} GFS 700 hPa Temperature Advection | {datetime_index[0].strftime("%Y-%m-%d %H00 UTC")} | FH: {time_diff_hours}', fontsize=14)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    directions = {'North': 51, 
                  'East': 250, 
                  'South': 20, 
                  'West': 200}
    g = 9.81
    (u, v, z, temp, mslp, pressure_levels, time) = function_a(directions)
    function_b(g, u, v, z, temp, mslp, pressure_levels, directions, time)
    function_c(g, u, v, z, temp, mslp, pressure_levels, directions, time)
