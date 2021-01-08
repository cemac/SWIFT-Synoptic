#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SWIFT automated synoptic charts

This module was developed by CEMAC as part of the GCRF African Swift
Project. This script allows automated plotting of synoptic features
across African domains.

.. module:: chart
   :synopsis: SWIFT automated synoptic charts

.. moduleauthor:: Tamora D. James <t.d.james1@leeds.ac.uk>, CEMAC (UoL)

:copyright: © 2020 University of Leeds.
:license: GPL 3.0 (see LICENSE)

Example::
        ./chart.py <domain> <forecast_timestamp> <forecast_hour> [<chart_type>]

        <domain> Domain specified as standardised domain name (WA or
                 EA)

        <timestamp> Timestamp for chart data in the format YYYYmmddHH

        <forecast_hour> Forecast hour as non-negative integer multiple
                 of 3 (max 72)

        <chart_type> Chart type (low, jets, conv or synth)

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)

import os
import sys
import datetime as dt
import argparse
import math
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as mcm
import matplotlib.colors as mc
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import iris
import iris.analysis.calculus
import numpy as np

#from * import gfs_utils  # works for tests but not when calling module code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import gfs_utils  # works when calling module code

class SynopticChart:
    """
    Synoptic chart

    """

    # Coords for standard domains from GFS_plotting/controls/domains
    # (could read in programatically from similar file)
    # Values are (min_lat, min_lon, max_lat, max_lon)
    # (could set up with value as {lat_min: -16.0, etc})

    # Domain specification is a tuple specifying (lat_min, lon_min,
    # lat_max, lon_max).
    DOMAINS = {
        'WA': (-2.5, -32.0, 36.0, 28.0),
        'EA': (-16.0, 20.0, 12.0, 50.0)
    }

    def __init__(self, domain, fct_timestamp, fct_hour, data_dir=None):
        self.set_domain(domain)
        self.set_timestamp(fct_timestamp)
        self.fct_hour = fct_hour
        self.set_data_dir(data_dir)
        self.components = []

        # Options for plotting backend
        self.mpl_options = {
            #frameon: False,
            'subplot_kw': { 'projection': ccrs.PlateCarree() },
        }

    def __str__(self):
        if self.chart_type is None:
            self.chart_type = "Synoptic"
        if self.domain_name is None:
            self.domain_name = "Custom"
        return f"{self.chart_type.capitalize()} chart for {self.get_date_str()}"
        " + {self.fct_hour} hours ({self.domain_name} domain)"

    def get_date_str(self):
        return dt.datetime.strptime(self.fct_timestamp, '%Y%m%d%H').strftime("%Y%m%d %H:%M")

    def set_domain(self, arg):
        arg_type = type(arg)
        if arg_type is str:
            if arg == 'WA' or arg == 'WestAfrica':
                self.domain = SynopticChart.DOMAINS['WA']
            elif arg == 'EA' or arg == 'EastAfrica':
                self.domain = SynopticChart.DOMAINS['EA']
            else:
                raise ValueError("Unrecognised domain name")
            self.domain_name = arg
        elif arg_type is dict:
            keys = ['lat_min', 'lon_min', 'lat_max', 'lon_max']
            if all([key in arg.keys() for key in keys]):
                self.domain = tuple([arg.get(key) for key in keys])
            else:
                raise ValueError("Domain does not contain required keys")
        elif arg_type is list or arg_type is tuple:
            if len(arg) == 4:
                self.domain = tuple(arg)
            else:
                raise ValueError("Domain does not contain the right number of values")
        else:
            raise ValueError("Unrecognised domain type")

    def set_timestamp(self, arg):
        try:
            if arg != dt.datetime.strptime(arg, '%Y%m%d%H').strftime('%Y%m%d%H'):
                raise ValueError('Unexpected timestamp format')
        except ValueError as e:
            raise(e)
        self.fct_timestamp = arg

    def set_data_dir(self, data_dir):
        if data_dir is None:
            # Use $SWIFT_GFS environment variable
            data_dir = os.getenv("SWIFT_GFS")
            if data_dir is None:
                raise RuntimeError('''Data directory not specified and
                $SWIFT_GFS environment variable is not set''')
        if not os.path.isdir(data_dir):
            raise ValueError("{:s} is not a directory".format(data_dir))
        self.data_dir = data_dir

    def date(self):
        return dt.datetime.strptime(self.fct_timestamp, '%Y%m%d%H')

    def get_file_path(self, timestamp):
        ts = dt.datetime.strptime(timestamp, '%Y%m%d%H')
        if self.fct_hour > 0:
            file_path = os.path.join(self.data_dir, "GFS_NWP", timestamp,
                                     'GFS_forecast_{:%Y%m%d_%H}.nc'.format(ts))
        else:
            file_path = os.path.join(self.data_dir, "GFS_NWP", timestamp,
                                     'analysis_gfs_4_{:%Y%m%d_%H}00_000.nc'.format(ts))
        if not os.path.isfile(file_path):
            raise ValueError("{:s} is not a file", file_path)
        return file_path

    def get_data(self, gfs_vars, units=None, delta=None):

        if delta is None:
            timestamp = self.fct_timestamp
        else:
            if not isinstance(delta, dt.timedelta):
                raise ValueError("Expecting datetime.timedelta object")

            ts = dt.datetime.strptime(self.fct_timestamp, "%Y%m%d%H") + delta
            timestamp = ts.strftime("%Y%m%d%H")

        # Load data
        with warnings.catch_warnings():
            # Suppress warnings
            warnings.simplefilter("ignore")
            cubes = iris.load_cubes(self.get_file_path(timestamp), gfs_vars)

        rtn = iris.cube.CubeList()
        for c in cubes:
            # Rewrap longitude
            c = c.intersection(longitude = (-180, 180))

            if units is not None:
                # Convert units
                c.convert_units(units)

            # Constrain to specified forecast hour for this chart
            fct_date = dt.datetime.strptime(c.attributes['initial_time'], '%m/%d/%Y (%H:%M)')
            time_constraint = gfs_utils.get_time_constraint(fct_date, self.fct_hour)
            c = c.extract(time_constraint)

            # Constrain to specified domain
            domain_constraint = gfs_utils.get_domain_constraint(self.domain)
            c = c.extract(domain_constraint)

            rtn.append(c)

        return rtn if len(rtn) > 1 else rtn[0]

    def add_component(self, component):
        self.components.append(component)

    def build(self, dir_path = None, scale_factor = 10):
        """Build chart from components"""
        if self.components:
            sample_data = self.components[0].data
            if isinstance(sample_data, iris.cube.CubeList):
                sample_data = sample_data[0]
            # Set sensible size in inches based on coordinate array extent
            coords = ('longitude', 'latitude')
            shape = np.array([sample_data.coord(x).points.size for x in coords])
            self.mpl_options['figsize'] = shape/scale_factor

        # set up chart - this might be configurable to use different
        # plotting back ends
        fig, ax = self.setup_plot()

        for c in self.components:
            # may need some concept of "order"?
            # add each component to plot
            c.plot(ax)

        if dir_path is not None:
            file_path = os.path.join(dir_path, '{:%Y%m%d%H%M}_{:02d}.png'.format(self.date(), 0))
            fig.savefig(file_path)
            plt.close()
        else:
            plt.show()

    def setup_plot(self):

        # Set up figure
        fig, ax = plt.subplots(**self.mpl_options)
        ax.set_position([0., 0., 1., 1.])  # Fill frame
        ax.set_axis_off()
        # Add coastlines and borders
        ax.coastlines(color='grey', alpha=0.8)
        ax.add_feature(cfeature.BORDERS, color='grey', alpha=0.5)
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, x_inline=True, y_inline=True, zorder=1.5)
        gl.xlocator = mticker.FixedLocator(np.arange(-30, 30, 10))
        gl.ylocator = mticker.FixedLocator(np.arange(0, 40, 10))
        gl.xlabel_style = { 'color': '#aaaaaa'}
        gl.ylabel_style = { 'color': '#aaaaaa'}

        # Add formatted date string
        date_str = '{:%Y%m%d %H:%M} + {:d} hours'.format(self.date(), self.fct_hour)
        ax.text(0, 0, date_str,
                horizontalalignment='left',
                verticalalignment='bottom',
                size='large',
                transform=ax.transAxes)

        return (fig, ax)

class LowLevelChart(SynopticChart):
    """Low-level chart displaying windspeed, streamlines, ITD based on
    dewpoint temperature, mean sea level pressure and pressure
    tendency.

    Note that this chart has been developed for West Africa. Extension
    to other domains may require some adjustments.

    """

    def __init__(self, domain, fct_timestamp, fct_hour, data_dir=None):
        super().__init__(domain, fct_timestamp, fct_hour, data_dir)

        self.chart_type = "low-level"

        # Mean sea level pressure
        self.mslp = MeanSeaLevelPressure(self)

        # Inter-tropical discontinuity
        self.itd = ITD(self)

        # Windspeed and streamlines at 925 hPa
        self.wc_925 = WindPressureLevel(self, 925)

        # 24 hour change in mean sea level pressure
        self.mslp_24 = MeanSeaLevelPressureChange(self)

        # Windspeed at 10m, 15 m/s contour
        self.wc_10m = WindHeightLevel(self, 10)

class WAJetsWaves(SynopticChart):
    """Chart displaying jets and waves for West Africa.

    Features to be plotted:
    * AEJ
    * STJ
    * TEJ
    * AEW troughs and ridges
    * PW* or Monsoon depth (to be renamed moisture depth) MD as filled contours
    * Monsoon trough [but this might be too difficult in practice]
    * Cyclonic centres at 850 hPa and 700/600 hPa level.
    * Possibly, dry air boundaries.

    """

    def __init__(self, domain, fct_timestamp, fct_hour, data_dir=None):
        super().__init__(domain, fct_timestamp, fct_hour, data_dir)

        self.chart_type = "WA-jets-waves"

        self.aej = AfricanEasterlyJet(self)

class SynopticComponent:
    """ Synoptic chart component """

    # GFS variables lookup
    GFS_VARS = {
        # dewpoint temperature at 2m
        'dpt_2m': 'DPT_P0_L103_GLL0',
        # precipitable water
        'pwat': 'PWAT_P0_L200_GLL0',
        # mean sea level pressure
        'prmsl': 'PRMSL_P0_L101_GLL0',
        # relative humidity
        'rh': 'RH_P0_L100_GLL0',
        # specific humidity, specified height level above ground ('lv_HTGL4')
        'spfh': 'SPFH_P0_L103_GLL0',
        # temperature on pressure levels
        'tmp': 'TMP_P0_L100_GLL0',
        # wind components on pressure levels
        'u_lvp': 'UGRD_P0_L100_GLL0',
        'v_lvp': 'VGRD_P0_L100_GLL0',
        # wind components at specific altitude (m) above MSL
        'u_lva': 'UGRD_P0_L102_GLL0',
        'v_lva': 'VGRD_P0_L102_GLL0',
        # wind components at specific height (m) above ground
        'u_lvh': 'UGRD_P0_L103_GLL0',
        'v_lvh': 'VGRD_P0_L103_GLL0',
        # surface pressure??
        'pres': 'PRES_P0_L1_GLL0',
    }

    def __init__(self, chart):
        self.init()
        self.chart = chart
        self.data = chart.get_data(self.gfs_vars, units=self.units)
        #self.backend = chart.get_backend()
        coords = ('latitude', 'longitude')
        if isinstance(self.data, iris.cube.CubeList):
            self.lat, self.lon = [self.data[0].coord(x).points for x in coords]
        else:
            self.lat, self.lon = [self.data.coord(x).points for x in coords]
        chart.add_component(self)

    def init():
        """
        Derived classes should implement this method to initialise
        self.gfs_vars and self.units, together with any class specific
        initialisation (e.g. formatting options).
        """
        raise RuntimeError("Not implemented")

    def plot(self, ax):
        # might change this to pass in chart then have a method in
        # chart that returns the plotting apparatus
        raise RuntimeError("Not implemented")

class MeanSeaLevelPressure(SynopticComponent):
    """
    Mean sea level pressure.
    """

    def __init__(self, chart):
        super().__init__(chart)

    def init(self):
        self.name = "Mean Sea Level Pressure"
        self.gfs_vars = SynopticComponent.GFS_VARS['prmsl']
        self.units = 'hPa'

        # Default to plot contour lines at 5 hPa intervals, 1015 bold
        self.step = 5
        self.highlight = 1015

        # Formatting options
        self.options = {
            'alpha': 0.6,
            'colors': 'purple',
        }

    def plot(self, ax):

        # Data to plot
        mslp = self.data.data

        pmin = np.amin(mslp)
        pmax = np.amax(mslp)
        levels = np.arange(pmin - pmin % self.step,
                           pmax + self.step - pmax % self.step,
                           self.step)
        lw = np.array([1.6 if x == self.highlight else 0.8 for x in levels])

        # Plot mean sea level pressure contours
        ctr = ax.contour(self.lon, self.lat, mslp,
                         levels=levels,
                         linewidths=lw,
                         **self.options)

class MeanSeaLevelPressureChange(MeanSeaLevelPressure):
    """
    Change in mean sea level pressure over 24 hours.
    """
    def __init__(self, chart):
        super().__init__(chart)

    def init(self):
        self.name = "Mean Sea Level Pressure Change"
        self.gfs_vars = SynopticComponent.GFS_VARS['prmsl']
        self.units = 'hPa'

        # Inspect MSLP change over past 24 hours
        self.delta = -24

        # Plot lines at 0.5 hPa intervals, skip contours in [-0.5, 0.5] interval
        self.step = 0.5
        self.thres = 0.5

        # Formatting options
        self.options = {
            'colors': 'orange',
        }

    def plot(self, ax):

        try:
            mslp_m24 = self.chart.get_data(self.gfs_vars, units=self.units,
                                           delta=dt.timedelta(hours=self.delta))
        except ValueError:
            msg = "{:d} hour data not available, cannot plot mean sea level pressure change".format(self.delta)
            warnings.warn(msg)
            return

        delta_mslp = self.data.data - mslp_m24.data

        dp_min = np.amin(delta_mslp)
        dp_max = np.amax(delta_mslp)
        levels = np.arange(dp_min - dp_min % self.step,
                           dp_max + self.step - dp_max % self.step,
                           self.step)
        levels = np.delete(levels, np.argwhere(np.abs(levels) <= self.thres))
        linestyles = np.where(levels < 0, '--', '-')

        ctr = ax.contour(self.lon, self.lat, delta_mslp,
                         levels=levels,
                         linestyles=linestyles,
                         **self.options)
        ax.clabel(ctr, fmt = '%1.1f')

class ITD(SynopticComponent):
    """
    Inter-tropical discontinuity (ITD), based on 15 deg C dewpoint
    temperature contour.

    """

    # could include options for type, masking etc.
    def __init__(self, chart):
        super().__init__(chart)

    def init(self):
        self.name = "Inter-tropical Discontinuity"
        self.gfs_vars = SynopticComponent.GFS_VARS['dpt_2m']
        self.units = 'Celsius'

        # Tolerance for gradient-based masking
        self.tol = 0.9
        self.dewpoint_level = [ 15.0 ]

        # Minimum number of vertices to include when plotting ITD contours
        self.min_vertices = 10

        # Formatting options
        self.options = {
        }

        # Plotting options
        self.lw = 1.0
        self.col = 'black'
        self.col2 = 'white'

    def plot(self, ax):

        # Calculate gradient of dewpoint temp with latitude
        gradient_dpt_wrt_lat = iris.analysis.calculus.differentiate(self.data, 'latitude')

        # Regrid dewpoint temperature to match gradient data
        dpt_2m_regridded = self.data.regrid(gradient_dpt_wrt_lat,
                                            iris.analysis.Linear(extrapolation_mode='extrapolate'))

        # Mask dpt_2m_regridded if gradient exceeds tolerance
        dpt_2m_masked = iris.util.mask_cube(dpt_2m_regridded,
                                            gradient_dpt_wrt_lat.data > self.tol)

        dewpoint = dpt_2m_masked.data
        self.lat = dpt_2m_masked.coord('latitude').points

        # Plot masked dewpoint temperature contour (ITD)
        itd1 = ax.contour(self.lon, self.lat, dewpoint,
                          levels = self.dewpoint_level,
                          colors = self.col, linewidths = 3*self.lw)
        itd2 = ax.contour(self.lon, self.lat, dewpoint,
                          levels = self.dewpoint_level,
                          colors = self.col2, linewidths = self.lw,
                          linestyles = 'dashed')

        # Clean up contours
        self.clean_contour(itd1)
        self.clean_contour(itd2)

    def clean_contour(self, ctr):
        """Clean up contour by getting rid of short paths"""
        for lc in ctr.collections:
            paths = lc.get_paths()
            new_paths = [path for path in paths if len(path.vertices) >= self.min_vertices]
            del paths[:]

            # set lc._paths directly because lc.set_paths() is buggy
            lc._paths = new_paths

class WindComponent(SynopticComponent):
    """
    Base class for wind-related synoptic components, providing utility
    method to extract U/V wind components and windspeed for a given level
    in the data.

    """

    def __init__(self, chart, level):
        super().__init__(chart)
        self.level = level

    def get_wind_components(self):
        """
        Extract U/V wind components and windspeed for a given level
        """
        # Get U/V wind components
        u, v = self.data.extract(self.gfs_vars)

        # Constrain to specified level
        u_lv_coord = gfs_utils.get_level_coord(u)
        if self.level_units is not None:
            u_lv_coord.convert_units(self.level_units)
        uc = gfs_utils.get_coord_constraint(u_lv_coord.name(), self.level)
        u = u.extract(uc)

        v_lv_coord = gfs_utils.get_level_coord(v)
        if self.level_units is not None:
            v_lv_coord.convert_units(self.level_units)
        vc = gfs_utils.get_coord_constraint(v_lv_coord.name(), self.level)
        v = v.extract(vc)

        U = u.data
        V = v.data
        windspeed = np.sqrt(U**2 + V**2)

        return (U, V, windspeed)

class WindPressureLevel(WindComponent):
    """
    Wind at specified pressure level

    """

    def __init__(self, chart, level):
        super().__init__(chart, level)

    def init(self):
        self.name = "Wind on Pressure Level"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x) for x in ('u_lvp', 'v_lvp')]
        self.units = None
        self.level_units = 'hPa'

        self.plot_ws = True
        self.plot_strm = True

        self.min_ws = 0
        self.max_ws = 50
        self.thres_ws = 10

        self.cm_name = 'Blues'

        # TODO create a util function to create the colour map
        # get_masked_colormap(name, min, max)
        cmap_hi = mcm.get_cmap(self.cm_name, 512)
        cmap = mc.ListedColormap(cmap_hi(np.linspace(0.1, 1.0, 256)),
                                   name=self.cm_name)
        cmap = cmap(np.linspace(0,1,256))
        cmap[:round(self.thres_ws/self.max_ws*256), :] = np.array([1, 1, 1, 1])

        # Formatting options
        self.ws_options = {
            'alpha': 0.4,
            'cmap': mc.ListedColormap(cmap),
        }

        self.strm_options = {
            'density': (2.5, 1.66),
            # 'color': 'black',
            # 'color': speed,
            # 'cmap': strm_cmap,
            'linewidth': 0.4,
            'arrowsize': 0.9,
            'arrowstyle': '->',
            # 'norm': cnorm,
        }

    def plot(self, ax):

        U, V, windspeed = self.get_wind_components()

        if self.plot_ws:
            # Plot windspeed at 925 hPa
            ctr = ax.contourf(self.lon, self.lat, windspeed,
                              **self.ws_options)

        if self.plot_strm:
            # Plot streamlines
            strm = ax.streamplot(self.lon, self.lat, U, V,
                                 **self.strm_options)

class WindHeightLevel(WindComponent):
    """
    Wind at specified height level

    """
    def __init__(self, chart, level):
        super().__init__(chart, level)

    def init(self):
        self.name = "Windspeed at Height Level"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x) for x in ('u_lvh', 'v_lvh')]
        self.units = None
        self.level_units = 'm'

        self.ws_level = [ 15.0 ]

        self.options = {
            'alpha': 0.6,
            'linewidths': 1.6,
            'colors': 'darkgreen',
        }

    def plot(self, ax):

        # Get U/V wind components
        _, _, windspeed = self.get_wind_components()

        # Plot 10m windspeed 15 m/s contour
        ctr = ax.contour(self.lon, self.lat, windspeed,
                         levels = self.ws_level,
                         **self.options)

class AfricanEasterlyJet(WindComponent):

    def __init__(self, chart, level=600):
        super().__init__(chart, level)

    def init(self):
        self.name = "African Easterly Jet"
        self.gfs_vars = [SynopticComponent.GFS_VARS.get(x) for x in ('u_lvp', 'v_lvp')]
        self.units = None
        self.level_units = 'hPa'

        self.plot_ws = True
        self.plot_strm = True

        self.min_ws = 0
        self.max_ws = 25
        self.thres_ws = 10

        self.cm_name = 'Greens'

        # TODO create a util function to create the colour map
        # get_masked_colormap(name, min, max)
        cmap_hi = mcm.get_cmap(self.cm_name, 512)
        cmap = mc.ListedColormap(cmap_hi(np.linspace(0.1, 1.0, 256)),
                                 name=self.cm_name)
        cmap = cmap(np.linspace(0,1,256))
        cmap[:round(self.thres_ws/self.max_ws*256), :] = np.array([1, 1, 1, 1])

        # Formatting options
        self.ws_options = {
            'alpha': 0.4,
            'cmap': mc.ListedColormap(cmap),
        }

        self.strm_options = {
            'color': 'green',
            # 'cmap': cmap,
            'linewidth': 0.8,
            'arrowsize': 1.9,
            'arrowstyle': '->',
            # 'norm': cnorm,
        }

    def plot(self, ax):

        U, V, windspeed = self.get_wind_components()

        if self.plot_ws:
            # Plot windspeed contours
            ctr = ax.contourf(self.lon, self.lat, windspeed,
                              **self.ws_options)

        if self.plot_strm:
            # Plot streamlines connecting maximum wind location

            # mask to relevant region i.e. between 10-15 deg N
            lat_grid = np.meshgrid(self.lon, self.lat)[1]
            mask = (lat_grid < 5) | (lat_grid > 20)

            # find max windspeed for the relevant region
            ws_masked = np.ma.masked_where(mask, windspeed)
            max_ws = np.amax(ws_masked)

            # select seed points
            seed_index = np.argwhere(ws_masked > max_ws*0.85)
            seed_points = np.array([[self.lon[x[1]], self.lat[x[0]]] for x in seed_index])

            # mask U, V to mask streamlines outside this area
            mask = ws_masked < self.thres_ws
            # U = np.ma.array(U, mask=mask)
            # V = np.ma.array(V, mask=mask)
            U[mask] = np.nan
            V[mask] = np.nan

            strm = ax.streamplot(self.lon, self.lat, U, V,
                                 start_points=seed_points,
                                 **self.strm_options)

            # TODO add arrowheads as quiver plot
            # ax.quiver(ax, ay, vx, vy, ...)

# class Template(SynopticComponent):

#     def __init__(self, chart):
#         super().__init__(chart)

#     def init(self):
#         self.name = "Name"
#         self.gfs_var = SynopticComponent.GFS_VARS['code']
#         self.units = None

#     def plot(self, ax):
#         pass

#---------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Plot synoptic chart')

    parser.add_argument('domain', type=str,
                        metavar='domain', choices=['WA', 'EA'],
                        help='''Domain specified as standardised
                        domain name (WA or EA)''')

    parser.add_argument('timestamp', type=str,
                        help='''Timestamp for chart data in format
                        \"YYYYmmddHH\"''')

    parser.add_argument('forecast_hour', type=int,
                        metavar='forecast_hour', choices=range(0,73,3),
                        help='''Forecast hour as non-negative integer
                        multiple of 3 (max 72)''')

    parser.add_argument('chart_type', nargs='?', type=str,
                        metavar="chart_type", choices=["low", "jets", "conv", "synth"],
                        default="low",
                        help='''Chart type (low, jets, conv or synth)''')

    pa = parser.parse_args()

    return (pa.domain, pa.timestamp, pa.forecast_hour, pa.chart_type)

def main():

    domain, timestamp, hour, chart_type = parse_args()

    if chart_type == "low":
        chart = LowLevelChart(domain, timestamp, hour)
    elif chart_type == "jets":
        chart = WAJetsWaves(domain, timestamp, hour)
    elif chart_type == "conv":
        chart = ConvectiveChart(domain, timestamp, hour)
    elif chart_type == "synth":
        chart = SynthesisChart(domain, timestamp, hour)
    else:
        raise ValueError("Unrecognised chart type")

    print(chart)

    chart.build()

    #end main()

if __name__ == '__main__':
    main()
