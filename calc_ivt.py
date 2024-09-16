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
    ivt_list = []
    int_time = u[time][0].values
    
    for i in range(0, 17): 
        u_i = u.isel(**{time: i})
        v_i = v.isel(**{time: i})
        q_i = q.isel(**{time: i})
        mslp_i = mslp.isel(**{time: i})
        time_values = u_i[time].values
        
        u_ij = u_i.sel(isobaric=pressure_levels)
        v_ij = v_i.sel(isobaric=pressure_levels)
        q_ij = q_i.sel(isobaric=pressure_levels)
        
        wnd_ij = mpcalc.wind_speed(u_ij, v_ij)
        
        ivt = -1 / g * np.trapz(wnd_ij * q_ij, pressure_levels, axis=0)
        u_ivt = -1 / g * np.trapz(u_ij * q_ij, pressure_levels, axis=0)
        v_ivt = -1 / g * np.trapz(v_ij * q_ij, pressure_levels, axis=0)
        
        ivt_da = xr.DataArray(ivt, dims=['lat', 'lon'], coords={'lat': u['lat'], 'lon': u['lon']}, name=f'IVT_time_{i}')
        ivt_list.append(ivt_da)
    
        int_datetime_index = pd.DatetimeIndex([int_time])
        datetime_index = pd.DatetimeIndex([time_values])
        time_diff_hours = int((datetime_index[0] - int_datetime_index[0]).total_seconds() / 3600)

        levels = [250, 300, 400, 500, 600, 700, 800, 1000, 1200, 1400, 1600, 1800]
        colors = ['#ffff00', '#ffe400', '#ffc800', '#ffad00', '#ff8200', '#ff5000', '#ff1e00', '#eb0010', '#b8003a', '#850063', '#570088']

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
        c = plt.contour(ivt_da['lon'], ivt_da['lat'], gaussian_filter(ivt_da, sigma=1), colors='black', levels=levels, linewidths=0.5)
        cf = plt.contourf(ivt_da['lon'], ivt_da['lat'], gaussian_filter(ivt_da, sigma=1), cmap=cmap, levels=levels, norm=norm)
        plt.colorbar(cf, extend='max', orientation='vertical', label='IVT (kg/m/s)', fraction=0.046, pad=0.04)

        mask = ivt_da >= 250
        u_ivt_filtered = xr.DataArray(u_ivt, dims=['lat', 'lon'], coords={'lat': u['lat'], 'lon': u['lon']}).where(mask, drop=True)
        v_ivt_filtered = xr.DataArray(v_ivt, dims=['lat', 'lon'], coords={'lat': u['lat'], 'lon': u['lon']}).where(mask, drop=True)

        step = 5 
        plt.quiver(u_ivt_filtered['lon'][::step], u_ivt_filtered['lat'][::step], u_ivt_filtered[::step, ::step], v_ivt_filtered[::step, ::step], scale=500,scale_units='xy', color='black')
        plt.title(f'{int_datetime_index[0].strftime("%H00 UTC")} GFS Integrated Vapor Transport (IVT) | {datetime_index[0].strftime("%Y-%m-%d %H00 UTC")} | FH: {time_diff_hours}', fontsize=14)
        plt.tight_layout()
        #plt.savefig(f'IVT_{i}.png', dpi=450)
        plt.show()


if __name__ == '__main__':

    directions = {'North': 51, 
                  'East': 250, 
                  'South': 20, 
                  'West': 220}
    g = 9.81
    (u, v, q, mslp, pressure_levels, time) = function_a(directions)
    function_b(g, u, v, q, mslp, pressure_levels, directions, time)
