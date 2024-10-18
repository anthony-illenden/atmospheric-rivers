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

def function_a(directions):
    tds_gfs = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/latest.html')
    gfs_ds = tds_gfs.datasets[0]
    ds = xr.open_dataset(gfs_ds.access_urls['OPENDAP'])
    ds_sliced = ds.sel(lat=slice(directions['North'], directions['South']), lon=slice(directions['West'], directions['East']))
    u = ds_sliced['u-component_of_wind_isobaric'].sel(isobaric=slice(30000, 100000))
    v = ds_sliced['v-component_of_wind_isobaric'].sel(isobaric=slice(30000, 100000))
    q = ds_sliced['Specific_humidity_isobaric'].sel(isobaric=slice(30000, 100000))
    mslp = ds_sliced['MSLP_Eta_model_reduction_msl']
    pressure_levels = u['isobaric'].values[::-1]
    time = time_dim(ds, 'u-component_of_wind_isobaric')
    return u, v, q, mslp, pressure_levels, time

def function_b(g, u, v, q, mslp, pressure_levels, directions, time):
    iwv_list = []
    int_time = u[time][0].values
    
    for i in range(0, 17): 
        u_i = u.isel(**{time: i})
        v_i = v.isel(**{time: i})
        q_i = q.isel(**{time: i})
        mslp_i = mslp.isel(**{time: i})
        time_values = u_i[time].values
        
        u_ij = u_i.sel(isobaric=pressure_levels)
        v_ij = v_i.sel(isobaric=pressure_levels)
        u_850 = u_i.sel(isobaric=85000)
        v_850 = v_i.sel(isobaric=85000)

        q_ij = q_i.sel(isobaric=pressure_levels)
        
        wnd_ij = mpcalc.wind_speed(u_ij, v_ij)
        
        iwv = -1 / g * np.trapz(q_ij, pressure_levels, axis=0)
        u_iwv = -1 / g * np.trapz(q_ij, pressure_levels, axis=0)
        v_iwv = -1 / g * np.trapz(q_ij, pressure_levels, axis=0)
        
        iwv_da = xr.DataArray(iwv, dims=['lat', 'lon'], coords={'lat': u['lat'], 'lon': u['lon']}, name=f'iwv_time_{i}')
        iwv_list.append(iwv_da)
    
        int_datetime_index = pd.DatetimeIndex([int_time])
        datetime_index = pd.DatetimeIndex([time_values])
        time_diff_hours = int((datetime_index[0] - int_datetime_index[0]).total_seconds() / 3600)

        levels = np.arange(20, 61, 2)
        colors = ['#1a2dd3', '#1a43ff', '#2486ff', '#31ccff', '#3cfbf0', '#37e5aa', '#32ce63', '#33be21', '#76d31c', '#bae814', '#fffc02',
                  '#ffe100', '#fec600', '#fdab00', '#fc7800', '#fc4100', '#fc0000','#d2002f', '#a31060', '#711e8b', '#8a51af']

        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)

        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([directions['West'], directions['East'], directions['South'], directions['North']-1])
        ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, color='#ecf9fd')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAKES, zorder=1, color='#ecf9fd')
        ax.add_feature(cfeature.LAND, color='#fbf5e9')
        gls = ax.gridlines(draw_labels=True, color='black', linestyle='--', alpha=0.35)
        gls.top_labels = False
        gls.right_labels = False

        c = plt.contour(mslp_i['lon'], mslp_i['lat'], gaussian_filter(mslp_i / 100, sigma=1), colors='black', levels=np.arange(960, 1040, 4))
        ax.clabel(c, levels=np.arange(980, 1020, 4), inline=True, inline_spacing=5, fontsize=10, fmt='%i')
        #c = plt.contour(iwv_da['lon'], iwv_da['lat'], gaussian_filter(iwv_da, sigma=1), colors='black', levels=levels, linewidths=0.5)
        cf = plt.contourf(iwv_da['lon'], iwv_da['lat'], gaussian_filter(iwv_da, sigma=1), cmap=cmap, levels=levels, norm=norm)
        plt.colorbar(cf, extend='max', orientation='vertical', label='IWV (mm)', fraction=0.046, pad=0.04)

        step = 5 
        ax.barbs(u_850['lon'][::step], u_850['lat'][::step], u_850[::step, ::step], v_850[::step, ::step], length=6, color='black')
        plt.title(f'{int_datetime_index[0].strftime("%H00 UTC")} GFS Integrated Water Vapor (IWV) | {datetime_index[0].strftime("%Y-%m-%d %H00 UTC")} | FH: {time_diff_hours}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'iwv_{i}.png', dpi=450)
        plt.show()


if __name__ == '__main__':

    directions = {'North': 51, 
                  'East': 250, 
                  'South': 20, 
                  'West': 220}
    g = 9.81
    (u, v, q, mslp, pressure_levels, time) = function_a(directions)
    function_b(g, u, v, q, mslp, pressure_levels, directions, time)
